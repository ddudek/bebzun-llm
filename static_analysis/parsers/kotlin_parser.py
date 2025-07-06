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
    
    def parse_file(self, file_path: Path, input_dir: Path) -> List[ClassStructure]:
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
            
            return self.extract_class_info(file_path, input_dir, file_content)
        except Exception as e:
            print(f"Error parsing Kotlin file {file_path}: {str(e)}")
            return []
    
    def extract_class_info(self, file_path: Path, input_dir: Path, file_content: str) -> List[ClassStructure]:
        """
        Extract class information from a Kotlin source file.
        
        Args:
            file_path: Path to the Kotlin source file
            file_content: Content of the file
            
        Returns:
            List of ClassSummaryOutput objects representing classes found in the file
        """
        classes = []
        
        # Parse the file content with tree-sitter
        tree = self._parse_with_tree_sitter(file_content)
        if not tree:
            return classes
        
        # Extract package name from the file
        package_name = self._extract_package_name(tree.root_node, file_content)
        
        # Find all class declarations (regular classes and data classes)
        class_nodes = []
        class_nodes.extend(self._find_nodes_by_type(tree.root_node, 'class_declaration'))
        class_nodes.extend(self._find_nodes_by_type(tree.root_node, 'object_declaration'))
        
        # Extract known classes for dependency resolution
        known_classes = set()
        for node in class_nodes:
            # In Kotlin AST, the class name is directly under class_declaration as 'identifier'
            for child in node.children:
                if child.type == 'identifier':
                    class_name = self._get_node_text(child, file_content)
                    known_classes.add(class_name)
                    break
        
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
            full_classname = f"{package_name}.{simple_classname}" if package_name else simple_classname
            
            # Extract dependencies for this class
            dependencies = self.extract_dependencies(file_content, simple_classname, known_classes)
            dependency_structs = []
            for dep_simple, dep_full, usage_lines in dependencies:
                dependency_structs.append(ClassStructureDependency(
                    simple_classname=dep_simple,
                    full_classname=dep_full,
                    usage_lines=usage_lines
                ))
            
            # Extract public methods for this class
            methods = self.extract_methods(file_content, simple_classname)
            method_structs = []
            for method_name, start_line, end_line in methods:
                method_structs.append(ClassStructureMethod(
                    name=method_name,
                    definition_start=start_line,
                    definition_end=end_line
                ))
            
            relative_path = file_path.relative_to(input_dir)

            # Create ClassSummaryOutput
            class_summary = ClassStructure(
                simple_classname=simple_classname,
                full_classname=full_classname,
                dependencies=dependency_structs,
                public_methods=method_structs,
                source_file=str(relative_path)
            )
            classes.append(class_summary)
        
        return classes
    
    def extract_dependencies(
        self,
        file_content: str,
        class_name: str,
        known_classes: Set[str]
    ) -> List[Tuple[str, str, List[int]]]:
        """
        Extract dependencies (classes used) from a Kotlin class.
        
        Args:
            file_content: Content of the file
            class_name: Name of the class to analyze
            known_classes: Set of class names that exist in the codebase
            
        Returns:
            List of tuples containing (simple_name, full_name, usage_lines)
        """
        dependencies = {}
        
        # Parse the file content
        tree = self._parse_with_tree_sitter(file_content)
        if not tree:
            return []
        
        # Extract imports to map simple names to full names
        imports = self._extract_imports(tree.root_node, file_content)
        
        # Find the target class node
        class_node = self._find_class_node(tree.root_node, class_name, file_content)
        if not class_node:
            return []
        
        # Track dependencies by walking the class AST recursive, keeping the scope
        # Build symbol table for scope
        parent_symbol_table = {}
        self._look_for_dependencies_scope_recursive(class_node, file_content, known_classes, imports, dependencies, parent_symbol_table)
        
        # Remove self-references
        if class_name in dependencies:
            del dependencies[class_name]
        
        # Convert to list format
        result = []
        for dep_name, dep_info in dependencies.items():
            result.append((dep_name, dep_info['full_name'], sorted(list(dep_info['lines']))))
        
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
        class_body = self._find_child_by_type(class_node, 'class_body')
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
    
    def _get_node_text(self, node: tree_sitter.Node, file_content: str) -> str:
        """Extract text content of a node."""
        return file_content[node.start_byte:node.end_byte]
    
    def _find_child_by_type(self, node: tree_sitter.Node, node_type: str) -> Optional[tree_sitter.Node]:
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
        class_nodes = self._find_nodes_by_type(root_node, 'class_declaration')
        class_nodes.extend(self._find_nodes_by_type(root_node, 'object_declaration'))
        
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
        known_classes: Set[str],
        symbol_table_out: Dict[str, str],
    ):
        for child in node.children:
            if child.type not in self.new_scope_node_types:
                self._build_symbol_table_current_scope_recursive(child, file_content, known_classes, symbol_table_out)
        return
    
    def _build_symbol_table_current_scope_recursive(
        self,
        node: tree_sitter.Node,
        file_content: str,
        known_classes: Set[str],
        symbol_table_out: Dict[str, str],
    ):
        """Build a symbol table mapping variable names to their types."""

        # if new scope, stop
        if node.type in self.new_scope_node_types:
            return

        # Handle primary constructor parameters
        if node.type == 'class_parameter':
            param_name_node = self._find_child_by_type(node, 'identifier')
            param_type_node = self._find_node_by_type(node, 'user_type')
            if param_name_node and param_type_node:
                param_name = self._get_node_text(param_name_node, file_content)
                type_id = self._find_node_by_type(param_type_node, 'identifier')
                if type_id:
                    type_name = self._get_node_text(type_id, file_content)
                    if type_name in known_classes:
                        if self.verbose:
                            print(f"[DEBUG] Found class param - {param_name}: {type_name}")
                        symbol_table_out[param_name] = type_name
        
        # Handle property declarations
        if node.type == 'property_declaration':
            var_decl = self._find_child_by_type(node, 'variable_declaration')
            type_node = self._find_node_by_type(node, 'user_type')
            if var_decl and type_node:
                var_id = self._find_child_by_type(var_decl, 'identifier')
                type_id = self._find_node_by_type(type_node, 'identifier')
                if var_id and type_id:
                    var_name = self._get_node_text(var_id, file_content)
                    type_name = self._get_node_text(type_id, file_content)
                    if type_name in known_classes:
                        if self.verbose:
                            print(f"[DEBUG] Found property - {var_name}: {type_name}")
                        symbol_table_out[var_name] = type_name
        
        # Handle function parameters
        if node.type == 'function_value_parameters':
            for param in self._find_nodes_by_type(node, 'parameter'):
                param_name_node = self._find_child_by_type(param, 'identifier')
                param_type_node = self._find_node_by_type(param, 'user_type')
                if param_name_node and param_type_node:
                    param_name = self._get_node_text(param_name_node, file_content)
                    type_id = self._find_node_by_type(param_type_node, 'identifier')
                    if type_id:
                        type_name = self._get_node_text(type_id, file_content)
                        if type_name in known_classes:
                            if self.verbose:
                                print(f"[DEBUG] Found function param - {param_name}: {type_name}")
                            symbol_table_out[param_name] = type_name
        
        # Recurse into children
        for child in node.children:
            if child.type not in self.new_scope_node_types:
                self._build_symbol_table_current_scope_recursive(child, file_content, known_classes, symbol_table_out)
    
    def _look_for_dependencies_scope_recursive(
        self,
        node: tree_sitter.Node,
        file_content: str,
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
            self._build_symbol_table_current_scope(node, file_content, known_classes, local_symbol_table)

        # Look for type references
        if node.type == 'user_type':
            identifier_node = self._find_child_by_type(node, 'identifier')
            if identifier_node:
                type_name = self._get_node_text(identifier_node, file_content)
                if type_name in known_classes:
                    line_number = node.start_point[0] + 1
                    if type_name not in dependencies_output:
                        full_name = imports.get(type_name, type_name)
                        dependencies_output[type_name] = {
                            'full_name': full_name,
                            'lines': set()
                        }
                    dependencies_output[type_name]['lines'].add(line_number)
        
        # Look for constructor calls
        elif node.type == 'call_expression':
            if node.children:
                first_child = node.children[0]
                if first_child.type in ['simple_identifier', 'identifier']:
                    identifier_name = self._get_node_text(first_child, file_content)
                    if identifier_name in known_classes:
                        line_number = node.start_point[0] + 1
                        if identifier_name not in dependencies_output:
                            full_name = imports.get(identifier_name, identifier_name)
                            dependencies_output[identifier_name] = {
                                'full_name': full_name,
                                'lines': set()
                            }
                        dependencies_output[identifier_name]['lines'].add(line_number)
                elif first_child.type == 'navigation_expression':
                    # Handle method calls like dep1.getName()
                    self._handle_navigation_expression(first_child, file_content, known_classes, imports,
                                                     dependencies_output, local_symbol_table)
        
        # Look for property access
        elif node.type == 'navigation_expression':
            self._handle_navigation_expression(node, file_content, known_classes, imports,
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
                var_type = self._resolve_variable_type(var_name, parent_symbol_table)
                if var_type and var_type in known_classes:
                    line_number = node.start_point[0] + 1
                    if var_type not in dependencies_output:
                        full_name = imports.get(var_type, var_type)
                        dependencies_output[var_type] = {
                            'full_name': full_name,
                            'lines': set()
                        }
                    dependencies_output[var_type]['lines'].add(line_number)
        
        # Look for string templates
        elif node.type == 'string_literal':
            # Look for interpolations within string templates
            for child in node.children:
                if child.type == 'interpolation':
                    # Track dependencies in interpolation expressions
                    self._look_for_dependencies_scope_recursive(child, file_content, known_classes, imports,
                                                        dependencies_output, parent_symbol_table)
        
        # Look for variable declarations with initializers
        elif node.type == 'property_declaration':
            # Get the variable name
            var_decl = self._find_child_by_type(node, 'variable_declaration')
            if var_decl:
                var_id = self._find_child_by_type(var_decl, 'identifier')
                if var_id:
                    var_name = self._get_node_text(var_id, file_content)
                    # Look for initializer
                    for child in node.children:
                        if child.type == 'call_expression':
                            # Constructor call in initializer
                            if child.children and child.children[0].type in ['simple_identifier', 'identifier']:
                                type_name = self._get_node_text(child.children[0], file_content)
                                if type_name in known_classes:
                                    line_number = child.start_point[0] + 1
                                    if type_name not in dependencies_output:
                                        full_name = imports.get(type_name, type_name)
                                        dependencies_output[type_name] = {
                                            'full_name': full_name,
                                            'lines': set()
                                        }
                                    dependencies_output[type_name]['lines'].add(line_number)
                                    # Update symbol table dynamically
                                    parent_symbol_table[var_name] = type_name
        
        # Recurse into children, including companion objects
        for child in node.children:
            self._look_for_dependencies_scope_recursive(child, file_content, known_classes, imports,
                                                dependencies_output, local_symbol_table)
    
    def _resolve_variable_type(self, var_name: str, symbol_table: Dict[str, str]) -> Optional[str]:
        """Resolve the type of a variable using the symbol table."""
        # Fall back to class-level lookup
        return symbol_table.get(var_name)
    
    def _handle_navigation_expression(
        self,
        node: tree_sitter.Node,
        file_content: str,
        known_classes: Set[str],
        imports: Dict[str, str],
        dependencies: Dict[str, Dict[str, any]],
        symbol_table: Dict[str, str]
    ):
        """Handle navigation expressions like variable.method() or variable.property."""
        if node.children:
            receiver = node.children[0]
            if receiver.type == 'simple_identifier':
                var_name = self._get_node_text(receiver, file_content)
                
                # Try function-scoped lookup first
                var_type = None
                
                # Fall back to class-level lookup
                if not var_type:
                    var_type = symbol_table.get(var_name)
                
                if var_type and var_type in known_classes:
                    line_number = node.start_point[0] + 1
                    if var_type not in dependencies:
                        full_name = imports.get(var_type, var_type)
                        dependencies[var_type] = {
                            'full_name': full_name,
                            'lines': set()
                        }
                    dependencies[var_type]['lines'].add(line_number)
    
    def _is_private_method(self, func_node: tree_sitter.Node, file_content: str) -> bool:
        """Check if a method is private."""
        # Look for modifiers
        modifiers = self._find_child_by_type(func_node, 'modifiers')
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