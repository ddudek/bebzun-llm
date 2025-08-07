"""
Kotlin source code parser implementation.

This module provides a Kotlin parser that uses tree-sitter AST parsing to extract
class information, methods, and dependencies. Tree-sitter is required for operation.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_kotlin as tskotlin

from static_analysis.model.model import ClassStructure, ClassStructureDependency, ClassStructureMethod
from static_analysis.parsers.base_parser import BaseParser

class KotlinParser(BaseParser):

    """
    Parser for Kotlin source files using tree-sitter AST parsing.
    
    This parser extracts class information, methods, and dependencies from Kotlin source files
    using tree-sitter AST. Tree-sitter is required for operation.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize the Kotlin parser with tree-sitter language grammar."""
        self.verbose = verbose
        self.new_scope_node_types = ["class_declaration", "function_declaration", "companion_object"]

        try:
            # Initialize the Kotlin language grammar
            self.kotlin_language = Language(tskotlin.language())
            
            # Create tree-sitter parser
            self.tree_sitter_parser = Parser()
            self.tree_sitter_parser.language = self.kotlin_language
            
            if self.verbose:
                print("[DEBUG] Tree-sitter Kotlin parser initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tree-sitter parser: {e}")
    
    def _parse_with_tree_sitter(self, content: str) -> Optional[tree_sitter.Tree]:
        """Parse source code with tree-sitter and return the AST."""
        try:
            content_bytes = content.encode('utf-8')
            tree = self.tree_sitter_parser.parse(content_bytes)
            return tree
        except Exception as e:
            raise RuntimeError(f"Tree-sitter parsing failed: {e}")
    
    def _find_nodes_by_type(self, node: tree_sitter.Node, node_type: str) -> List[tree_sitter.Node]:
        """Recursively find all nodes of a specific type in the AST."""
        if not node:
            return []
        
        nodes = []
        if node.type == node_type:
            nodes.append(node)
        
        for child in node.children:
            nodes.extend(self._find_nodes_by_type(child, node_type))
        
        return nodes
    

    def _find_node_by_type(self, node: tree_sitter.Node, node_type: str) -> Optional[tree_sitter.Node]:
        """Recursively find all nodes of a specific type in the AST."""
        if not node:
            return None
        
        if node and node.type == node_type:
            return node
        
        for child in node.children:
            child_node = self._find_node_by_type(child, node_type)
            if child_node:
                return child_node
        
        return None
    
    def extract_classes(self, file_path: Path, input_dir: Path, version: int) -> List[ClassStructure]:
        """
        Parse a Kotlin source file and extract class information.
        
        Args:
            file_path: Path to the Kotlin source file
            
        Returns:
            List of ClassSummaryOutput objects representing classes found in the file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()

            relative_path = file_path.relative_to(input_dir)

            classes = []

            # Parse the file content with tree-sitter
            tree = self._parse_with_tree_sitter(file_content)
            if not tree:
                return classes
            
            # Extract package name from the file
            package_name = self._extract_package_name(tree.root_node, file_content)
            if not package_name:
                print(f"Error: can't find package name for {file_path}")
                return classes
            
            # Find all class declarations (regular classes and data classes)
            class_nodes = self._find_class_nodes_in_file(tree.root_node)
            
            # Process each class
            for class_node in class_nodes:
                # Find the class name - it's a direct child identifier
                class_name_node = None
                for child in class_node.children:
                    if child.type == 'identifier':
                        class_name_node = child
                        break
                
                if not class_name_node:
                    continue
                
                simple_classname = self._get_node_text(class_name_node, file_content)

                # nested classes eg. SomeState.Success
                simple_classname_nested = self._find_nested_classname(simple_classname, class_node)

                full_classname = f"{package_name}.{simple_classname_nested}"

                # Create ClassSummaryOutput
                class_summary = ClassStructure(
                    simple_classname=simple_classname,
                    full_classname=full_classname,
                    dependencies=[],
                    public_methods=[],
                    source_file=str(relative_path),
                    version=version
                )
                classes.append(class_summary)
            
            return classes
            
        except Exception as e:
            print(f"Error parsing Kotlin file {file_path}: {str(e)}")
            return []
        
    def _find_nested_classname(self, simple_classname, class_node):
        if not class_node.parent or class_node == class_node.parent:
            return simple_classname

        if class_node.parent and class_node.parent.parent and class_node.parent.parent.type == 'class_declaration':
            parent_class_node = class_node.parent.parent
            for child in parent_class_node.children:
                if child.type == 'identifier':
                    simple_classname = f"{self._get_node_text(child)}.{simple_classname}"
                    simple_classname = self._find_nested_classname(simple_classname, parent_class_node)
                    break

        return simple_classname
    
    def _find_class_nodes_in_file(self, root_node: tree_sitter.Node) -> List[tree_sitter.Node]:
        class_nodes = []
        class_nodes.extend(self._find_nodes_by_type(root_node, 'class_declaration'))
        class_nodes.extend(self._find_nodes_by_type(root_node, 'object_declaration'))

        compose_fun_nodes = self._find_nodes_by_type(root_node, 'function_declaration')
        for fun_node in compose_fun_nodes:
            modifiers = self._find_nodes_by_type(fun_node, 'modifiers')
            for modifier in modifiers:
                modifiers_text = self._get_node_text(modifier)
                if 'Composable' in modifiers_text and 'Preview' not in modifiers_text:
                    class_name = None
                    for child in fun_node.children:
                        if child.type == 'identifier':
                            class_name = self._get_node_text(child)
                            if (class_name[0].isupper()):
                                class_nodes.append(fun_node)
                            break
                    break
                
        return class_nodes
    
    def extract_dependencies(
        self,
        file_content: str,
        class_structure: ClassStructure,
        known_classes: Set[str]
    ) -> List[Tuple[str, List[int]]]:
        """
        Extract dependencies (classes used) from a Kotlin class.
        
        Args:
            file_content: Content of the file
            class_name: Name of the class to analyze
            known_classes: Set of class names that exist in the codebase
            
        Returns:
            List of tuples containing (simple_name, full_name, usage_lines)
        """
        dependencies: Dict[str, Dict[str, any]] = {}
        
        # Parse the file content
        tree = self._parse_with_tree_sitter(file_content)
        if not tree:
            return []
        
        # Extract imports to map simple names to full names
        imports = self._extract_imports(tree.root_node, file_content)
        
        package_name = self._extract_package_name(tree.root_node, file_content)
        if not package_name:
            print(f"Warning: can't find package name for {class_structure.full_classname}")
            return []

        # Find the target class node
        class_node = self._find_class_node(tree.root_node, class_structure.simple_classname, file_content)
        if not class_node:
            return []
        
        # Track dependencies by walking the class AST recursive, keeping the scope
        # Build symbol table for scope
        self._look_for_dependencies_scope_recursive(class_node, file_content, package_name, known_classes, imports, dependencies, {})
        
        # Convert to list format
        result = []
        for dep_name, dep_info in dependencies.items():
            if class_structure.full_classname != dep_name: # Remove self-references
                result.append((dep_info['full_name'], sorted(list(dep_info['lines']))))
        
        return result
    
    def extract_methods(
        self,
        file_content: str,
        class_name: str
    ) -> List[Tuple[str, int, int]]:
        """
        Extract public methods from a Kotlin class.
        
        Args:
            file_content: Content of the file
            class_name: Name of the class to analyze
            
        Returns:
            List of tuples containing (method_name, start_line, end_line)
        """
        methods = []
        
        # Parse the file content
        tree = self._parse_with_tree_sitter(file_content)
        if not tree:
            return methods
        
        # Find the target class node
        class_node = self._find_class_node(tree.root_node, class_name, file_content)
        if not class_node:
            return methods
        
        # Find the class body
        class_body = self._find_direct_child_by_type(class_node, 'class_body')
        if not class_body:
            return methods
        
        # Find all function declarations in the class body
        function_nodes = self._find_nodes_by_type(class_body, 'function_declaration')
        
        for func_node in function_nodes:
            # Skip if inside companion object
            if self._is_inside_companion_object(func_node):
                continue
            
            # Check visibility (skip private methods)
            if self._is_private_method(func_node, file_content):
                continue
            
            # Get method name using field name
            name_node = self._find_child_by_field(func_node, 'name')
            if not name_node:
                continue
            
            method_name = self._get_node_text(name_node, file_content)
            
            # Get line numbers
            start_line = func_node.start_point[0] + 1  # tree-sitter uses 0-based indexing
            end_line = func_node.end_point[0] + 1
            
            methods.append((method_name, start_line, end_line))
        
        return methods
    
    # Helper methods
    
    def _get_node_text(self, node: tree_sitter.Node, file_content: str = "") -> str:
        """Extract text content of a node."""
        return str(node.text, encoding='utf-8')
    
    def _find_direct_child_by_type(self, node: tree_sitter.Node, node_type: str) -> Optional[tree_sitter.Node]:
        """Find the first child node of a specific type."""
        for child in node.children:
            if child.type == node_type:
                return child
        return None
    
    def _find_child_by_field(self, node: tree_sitter.Node, field_name: str) -> Optional[tree_sitter.Node]:
        """Find a child node by field name."""
        return node.child_by_field_name(field_name)
    
    def _extract_package_name(self, root_node: tree_sitter.Node, file_content: str) -> str:
        """Extract the package name from the file."""
        package_nodes = self._find_nodes_by_type(root_node, 'package_header')
        if package_nodes:
            package_node = package_nodes[0]
            # Look for qualified_identifier
            qualified_id_nodes = self._find_nodes_by_type(package_node, 'qualified_identifier')
            if qualified_id_nodes:
                qualified_id = qualified_id_nodes[0]
                identifier_nodes = self._find_nodes_by_type(qualified_id, 'identifier')
                if identifier_nodes:
                    parts = []
                    for id_node in identifier_nodes:
                        parts.append(self._get_node_text(id_node, file_content))
                    return '.'.join(parts)
        return ''
    
    def _find_class_node(self, root_node: tree_sitter.Node, class_name: str, file_content: str) -> Optional[tree_sitter.Node]:
        """Find a class node by name."""
        class_nodes = self._find_class_nodes_in_file(root_node)
        
        for node in class_nodes:
            # The class name is a direct child identifier
            for child in node.children:
                if child.type == 'identifier' and self._get_node_text(child, file_content) == class_name:
                    return node
        return None
    
    def _extract_imports(self, root_node: tree_sitter.Node, file_content: str) -> Dict[str, str]:
        """Extract import statements to map simple names to full names."""
        imports = {}
        import_nodes = self._find_nodes_by_type(root_node, 'import')
        
        for import_node in import_nodes:
            # Look for qualified_identifier
            qualified_id_nodes = self._find_nodes_by_type(import_node, 'qualified_identifier')
            if qualified_id_nodes:
                qualified_id = qualified_id_nodes[0]
                identifier_nodes = self._find_nodes_by_type(qualified_id, 'identifier')
                if identifier_nodes:
                    parts = [self._get_node_text(node, file_content) for node in identifier_nodes]
                    if parts:
                        simple_name = parts[-1]
                        full_name = '.'.join(parts)
                        imports[simple_name] = full_name
        
        return imports
    
    def _build_symbol_table_current_scope(
        self,
        node: tree_sitter.Node,
        file_content: str,
        package: str,
        known_classes: Set[str],
        imports: Dict[str, str],
        symbol_table_out: Dict[str, str],
    ):
        for child in node.children:
            if child.type not in self.new_scope_node_types:
                self._build_symbol_table_current_scope_recursive(child, file_content, package, known_classes, imports, symbol_table_out)
        return
    
    def _build_symbol_table_current_scope_recursive(
        self,
        node: tree_sitter.Node,
        file_content: str,
        package: str,
        known_classes: Set[str],
        imports: Dict[str, str],
        symbol_table_out: Dict[str, str],
    ):
        """Build a symbol table mapping variable names to their types."""

        # if new scope, stop
        if node.type in self.new_scope_node_types:
            return

        # Handle primary constructor parameters
        if node.type == 'class_parameter':
            param_name_node = self._find_direct_child_by_type(node, 'identifier')
            param_type_node = self._find_node_by_type(node, 'user_type')
            if param_name_node and param_type_node:
                param_name = self._get_node_text(param_name_node, file_content)
                type_id = self._find_node_by_type(param_type_node, 'identifier')
                if type_id:
                    type_name = self._get_node_text(type_id, file_content)
                    full_name = imports.get(type_name, f"{package}.{type_name}")
                    if full_name in known_classes:
                        if self.verbose:
                            print(f"[DEBUG] Found class param - {param_name}: {type_name}")
                        symbol_table_out[param_name] = full_name
        
        # Handle property declarations
        if node.type == 'property_declaration':
            var_decl = self._find_direct_child_by_type(node, 'variable_declaration')
            type_node = self._find_node_by_type(node, 'user_type')
            if not type_node:
                type_node = self._find_node_by_type(node, 'call_expression')
            if var_decl and type_node:
                var_id = self._find_direct_child_by_type(var_decl, 'identifier')
                type_id = self._find_node_by_type(type_node, 'identifier')
                if var_id and type_id:
                    var_name = self._get_node_text(var_id, file_content)
                    type_name = self._get_node_text(type_id, file_content)
                    full_name = imports.get(type_name, f"{package}.{type_name}")
                    if full_name in known_classes:
                        if self.verbose:
                            print(f"[DEBUG] Found property - {var_name}: {type_name}")
                        symbol_table_out[var_name] = full_name
        
        # Handle function parameters
        if node.type == 'function_value_parameters':
            for param in self._find_nodes_by_type(node, 'parameter'):
                param_name_node = self._find_direct_child_by_type(param, 'identifier')
                param_type_node = self._find_node_by_type(param, 'user_type')
                if param_name_node and param_type_node:
                    param_name = self._get_node_text(param_name_node, file_content)
                    type_id = self._find_node_by_type(param_type_node, 'identifier')
                    if type_id:
                        type_name = self._get_node_text(type_id, file_content)
                        full_name = imports.get(type_name, f"{package}.{type_name}")
                        if full_name in known_classes:
                            if self.verbose:
                                print(f"[DEBUG] Found function param - {param_name}: {type_name}")
                            symbol_table_out[param_name] = full_name
        
        # Recurse into children
        for child in node.children:
            if child.type not in self.new_scope_node_types:
                self._build_symbol_table_current_scope_recursive(child, file_content, package, known_classes, imports, symbol_table_out)
    
    def _look_for_dependencies_scope_recursive(
        self,
        node: tree_sitter.Node,
        file_content: str,
        package: str,
        known_classes: Set[str],
        imports: Dict[str, str],
        dependencies_output: Dict[str, Dict[str, any]],
        parent_symbol_table: Dict[str, str],
    ):
        """Track dependencies"""
        
        # Create current_symbol_table: a copy of parent_symbol_table that will be updated with current symbols
        local_symbol_table = parent_symbol_table
        if node.type in self.new_scope_node_types:
            local_symbol_table = parent_symbol_table.copy()
            self._build_symbol_table_current_scope(node, file_content, package, known_classes, imports, local_symbol_table)

        # Look for type references
        if node.type == 'user_type':
            identifier_node = self._find_direct_child_by_type(node, 'identifier')
            if identifier_node:
                type_name = self._get_node_text(identifier_node, file_content)
                full_name = imports.get(type_name, f"{package}.{type_name}")
                if full_name in known_classes:
                    line_number = node.start_point[0] + 1
                    if full_name not in dependencies_output:
                        dependencies_output[full_name] = {
                            'full_name': full_name,
                            'lines': set()
                        }
                    dependencies_output[full_name]['lines'].add(line_number)
        
        # Look for constructor calls
        elif node.type == 'call_expression':
            if node.children:
                first_child = node.children[0]
                if first_child.type in ['simple_identifier', 'identifier']:
                    type_name = self._get_node_text(first_child, file_content)
                    full_name = imports.get(type_name, f"{package}.{type_name}")
                    if full_name in known_classes:
                        line_number = node.start_point[0] + 1
                        if full_name not in dependencies_output:
                            dependencies_output[full_name] = {
                                'full_name': full_name,
                                'lines': set()
                            }
                        dependencies_output[full_name]['lines'].add(line_number)
                elif first_child.type == 'navigation_expression':
                    # Handle method calls like dep1.getName()
                    self._handle_navigation_expression(first_child, file_content, package, known_classes, imports,
                                                     dependencies_output, local_symbol_table)
        
        # Look for property access
        elif node.type == 'navigation_expression':
            self._handle_navigation_expression(node, file_content, package, known_classes, imports,
                                             dependencies_output, local_symbol_table)
        
        # Look for simple identifier references (variables/parameters)
        elif node.type == 'simple_identifier' or node.type == 'identifier':
            # Check if it's not part of a navigation expression, call expression, or type reference
            parent = node.parent
            # Skip if it's the called function name, navigation target, or type reference
            skip_parents = [
                'navigation_expression',  # Skip if it's the right side of a dot
                'user_type',  # Skip if it's a type reference
                'parameter',  # Skip parameter names
                'property_declaration',  # Skip property names
                'variable_declaration',  # Skip variable names
                'function_declaration',  # Skip function names
                'class_declaration',  # Skip class names
            ]
            
            # Special handling for navigation expressions - only skip if it's not the receiver
            if parent and parent.type == 'navigation_expression':
                # If this identifier is the first child (receiver), don't skip
                if parent.children and parent.children[0] == node:
                    skip_parents = [p for p in skip_parents if p != 'navigation_expression']
            
            # For call expressions, only track if it's not the function being called
            if parent and parent.type == 'call_expression':
                # Skip if it's the first child (function name)
                if parent.children and parent.children[0] == node:
                    skip_parents.append('call_expression')
            
            if parent and parent.type not in skip_parents:
                var_name = self._get_node_text(node, file_content)
                full_name = self._resolve_variable_type(var_name, parent_symbol_table)
                if full_name and full_name in known_classes:
                    line_number = node.start_point[0] + 1
                    if full_name not in dependencies_output:                    
                        dependencies_output[full_name] = {
                            'full_name': full_name,
                            'lines': set()
                        }
                    dependencies_output[full_name]['lines'].add(line_number)
        
        # Look for string templates
        elif node.type == 'string_literal':
            # Look for interpolations within string templates
            for child in node.children:
                if child.type == 'interpolation':
                    # Track dependencies in interpolation expressions
                    self._look_for_dependencies_scope_recursive(child, file_content, package, known_classes, imports,
                                                        dependencies_output, parent_symbol_table)
        
        # Look for variable declarations with initializers
        elif node.type == 'property_declaration':
            # Get the variable name
            var_decl = self._find_direct_child_by_type(node, 'variable_declaration')
            if var_decl:
                var_id = self._find_direct_child_by_type(var_decl, 'identifier')
                if var_id:
                    var_name = self._get_node_text(var_id)
                    # Look for initializer
                    type_fullname = None
                    user_type = self._find_node_by_type(var_decl, 'user_type')
                    if user_type and user_type.children and user_type.children[0].type in ['simple_identifier', 'identifier']:
                        type_name = self._get_node_text(user_type.children[0])
                        type_fullname = imports.get(type_name, f"{package}.{type_name}")
                        type_node = user_type.children[0]
                    
                    delegate = self._find_node_by_type(node, 'property_delegate')
                    if delegate:
                        type_nodes = self._find_nodes_by_type(delegate, 'identifier')
                        for p_type_node in type_nodes:
                            potential_type_name = self._get_node_text(p_type_node)
                            potential_type_fullname = imports.get(potential_type_name, f"{package}.{potential_type_name}")
                            if potential_type_fullname in known_classes:
                                type_node = p_type_node
                                type_fullname = potential_type_fullname

                    for child in node.children:
                        if child.type == 'call_expression':
                            # Constructor call in initializer
                            if child.children and child.children[0].type in ['simple_identifier', 'identifier']:
                                type_name = self._get_node_text(child.children[0], file_content)
                                type_fullname = imports.get(type_name, f"{package}.{type_name}")
                                type_node = child.children[0]

                    if type_fullname and type_fullname in known_classes:
                        line_number = type_node.start_point[0] + 1
                        if type_fullname not in dependencies_output:
                            dependencies_output[type_fullname] = {
                                'full_name': type_fullname,
                                'lines': set()
                            }
                        dependencies_output[type_fullname]['lines'].add(line_number)
                        # Update symbol table dynamically
                        parent_symbol_table[var_name] = type_fullname
        
        # Recurse into children, including companion objects
        for child in node.children:
            self._look_for_dependencies_scope_recursive(child, file_content, package, known_classes, imports,
                                                dependencies_output, local_symbol_table)
    
    def _resolve_variable_type(self, var_name: str, symbol_table: Dict[str, str]) -> Optional[str]:
        """Resolve the type of a variable using the symbol table."""
        # Fall back to class-level lookup
        return symbol_table.get(var_name)
    
    def _handle_navigation_expression(
        self,
        node: tree_sitter.Node,
        file_content: str,
        package: str,
        known_classes: Set[str],
        imports: Dict[str, str],
        dependencies: Dict[str, Dict[str, any]],
        symbol_table: Dict[str, str]
    ):
        """Handle navigation expressions like variable.method() or variable.property."""
        if node.children:
            for receiver in node.named_children:
                if receiver.type == 'simple_identifier' or receiver.type == 'identifier':
                    var_name = self._get_node_text(receiver, file_content)
                    
                    var_name_as_type = imports.get(var_name, var_name)

                    var_type = symbol_table.get(var_name)

                    # if receiver is a type, e.g. ExampleClass.something
                    var_name_as_type = imports.get(var_name, var_name)
                    if var_name_as_type in known_classes:
                        var_type = var_name_as_type

                    # import com.example.SomeState
                    # SomeState.Initial
                    root_type = self._get_node_text(node.children[0])
                    expression_as_type = self._get_node_text(node)
                    if root_type in imports:
                        expression_as_full_type = imports.get(root_type).replace(root_type, expression_as_type)
                        if expression_as_full_type in known_classes:
                            var_type = expression_as_full_type

                    # no import
                    # package + SomeState.Initial
                    expression_as_full_type = f"{package}.{expression_as_type}"
                    if expression_as_full_type in known_classes:
                        var_type = expression_as_full_type
                    
                    if var_type and var_type in known_classes:
                        line_number = node.start_point[0] + 1
                        if var_type not in dependencies:
                            dependencies[var_type] = {
                                'full_name': var_type,
                                'lines': set()
                            }
                        dependencies[var_type]['lines'].add(line_number)
    
    def _is_private_method(self, func_node: tree_sitter.Node, file_content: str) -> bool:
        """Check if a method is private."""
        # Look for modifiers
        modifiers = self._find_direct_child_by_type(func_node, 'modifiers')
        if modifiers:
            for child in modifiers.children:
                if child.type == 'visibility_modifier':
                    modifier_text = self._get_node_text(child, file_content)
                    if modifier_text == 'private':
                        return True
        return False
    
    def _is_inside_companion_object(self, node: tree_sitter.Node) -> bool:
        """Check if a node is inside a companion object."""
        current = node.parent
        while current:
            if current.type == 'companion_object':
                return True
            current = current.parent
        return False