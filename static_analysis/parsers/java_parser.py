"""
Java source code parser implementation.

This module provides functionality for parsing Java source files using the tree-sitter library.
It extracts class information, methods, and dependencies.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava

from static_analysis.model.model import ClassStructure, ClassStructureDependency, ClassStructureMethod
from static_analysis.parsers.base_parser import BaseParser

class JavaParser(BaseParser):
    """
    Parser for Java source files using the tree-sitter library.

    This parser extracts class information, methods, and dependencies from Java source files.
    """

    def __init__(self, verbose: bool = False):
        """Initialize the Java parser with tree-sitter language grammar."""
        self.verbose = verbose
        self.new_scope_node_types = ["class_declaration", "interface_declaration", "method_declaration", "constructor_declaration"]
        try:
            self.java_language = Language(tsjava.language())
            self.tree_sitter_parser = Parser()
            self.tree_sitter_parser.language = self.java_language
            if self.verbose:
                print("[DEBUG] Tree-sitter Java parser initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tree-sitter parser for Java: {e}")

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

    def _get_node_text(self, node: tree_sitter.Node, file_content: str = "") -> str:
        """Extract text content of a node."""
        return str(node.text, encoding='utf-8')

    def _extract_package_name(self, root_node: tree_sitter.Node) -> str:
        """Extract the package name from the file."""
        package_nodes = self._find_nodes_by_type(root_node, 'package_declaration')
        if package_nodes:
            package_node = package_nodes[0]
            name_node = self._find_nodes_by_type(package_node, 'scoped_identifier')
            if not name_node:
                name_node = self._find_nodes_by_type(package_node, 'identifier')
            
            if name_node:
                return self._get_node_text(name_node[0])
        return ''

    def extract_classes(self, file_path: Path, input_dir: Path, version: int) -> List[ClassStructure]:
        """
        Parse a Java source file and extract class information.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()

            relative_path = file_path.relative_to(input_dir)
           
            classes = []
            tree = self._parse_with_tree_sitter(file_content)
            if not tree:
                return classes

            package_name = self._extract_package_name(tree.root_node)
            class_nodes = self._find_nodes_by_type(tree.root_node, 'class_declaration')
            interface_nodes = self._find_nodes_by_type(tree.root_node, 'interface_declaration')
            class_nodes.extend(interface_nodes)

            if not package_name:
                print(f"Error: can't find package name for {file_path}")
                return classes

            for class_node in class_nodes:
                name_node = class_node.child_by_field_name('name')
                if name_node:
                    simple_classname = self._get_node_text(name_node, file_content)

                    # nested classes eg. SomeState.Success
                    simple_classname_nested = self._find_nested_classname(simple_classname, class_node)

                    full_classname = f"{package_name}.{simple_classname_nested}"

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
            print(f"Error parsing Java file {file_path}: {str(e)}")
            return []
        
    def _find_nested_classname(self, simple_classname, class_node):
        if not class_node.parent or not class_node.parent.type == 'class_body' or class_node == class_node.parent:
            return simple_classname

        if class_node.parent and class_node.parent.parent and class_node.parent.parent.type == 'class_declaration':
            parent_class_node = class_node.parent.parent
            for child in parent_class_node.children:
                if child.type == 'identifier':
                    simple_classname = f"{self._get_node_text(child)}.{simple_classname}"
                    simple_classname = self._find_nested_classname(simple_classname, parent_class_node)
                    break

        return simple_classname

    def _extract_imports(self, root_node: tree_sitter.Node, file_content: str) -> Dict[str, str]:
        """
        Extract import statements to map simple names to fully qualified names.
        """
        imports = {}
        import_nodes = self._find_nodes_by_type(root_node, 'import_declaration')
        
        for import_node in import_nodes:
            name_node = None
            if not import_node.child_by_field_name('asterisk'):
                name_node_list = self._find_nodes_by_type(import_node, 'scoped_identifier')
                if not name_node_list:
                    name_node_list = self._find_nodes_by_type(import_node, 'identifier')
                if name_node_list:
                    name_node = name_node_list[0]

            if name_node:
                full_import_name = self._get_node_text(name_node, file_content)
                parts = full_import_name.split('.')
                if parts:
                    simple_name = parts[-1]
                    imports[simple_name] = full_import_name
                    
        return imports
        
    def _add_dependency(
        self,
        type_name: str,
        package: str,
        line_number: int,
        class_name: str,
        known_classes: Set[str],
        imports: Dict[str, str],
        dependencies: Dict[str, Dict[str, any]]
    ):
        """
        Add a dependency to the dependencies dict if it's a known class.
        """
        if not type_name or type_name == class_name:
            return
        
        full_name = imports.get(type_name, f"{package}.{type_name}")

        if full_name in known_classes:
            if full_name not in dependencies:
                dependencies[full_name] = {'full_name': full_name, 'lines': set()}
            dependencies[full_name]['lines'].add(line_number)
    
    def extract_dependencies(
        self,
        file_content: str,
        class_structure: ClassStructure,
        known_classes: Set[str]
    ) -> List[Tuple[str, List[int]]]:
        """
        Extract dependencies from a Java class.
        """
        dependencies = {}
        tree = self._parse_with_tree_sitter(file_content)
        if not tree:
            return []
            
        imports = self._extract_imports(tree.root_node, file_content)
        package = self._extract_package_name(tree.root_node)
        all_known_types = known_classes.union(set(imports.keys()))

        class_nodes = []
        class_nodes.extend(self._find_nodes_by_type(tree.root_node, 'class_declaration'))
        class_nodes.extend(self._find_nodes_by_type(tree.root_node, 'interface_declaration'))

        target_class_node = None
        for class_node in class_nodes:
            name_node = class_node.child_by_field_name('name')
            if name_node and self._get_node_text(name_node, file_content) == class_structure.simple_classname:
                target_class_node = class_node
                break
                
        if not target_class_node:
            return []
        
        parent_symbol_table = {}
        self._look_for_dependencies_scope_recursive(
            target_class_node, file_content, package, all_known_types, imports, dependencies, parent_symbol_table, class_structure.full_classname
        )
        
        if class_structure.full_classname in dependencies:
            del dependencies[class_structure.full_classname]
        
        result = []
        for dep_name, dep_info in dependencies.items():
            result.append((dep_info['full_name'], sorted(list(dep_info['lines']))))
            
        return result

    def _build_symbol_table_current_scope(
        self,
        node: tree_sitter.Node,
        file_content: str,
        known_classes: Set[str],
        imports: Dict[str, str],
        symbol_table_out: Dict[str, str],
    ):
        for child in node.children:
            if child.type not in self.new_scope_node_types:
                self._build_symbol_table_current_scope_recursive(child, file_content, known_classes, imports, symbol_table_out)

    def _build_symbol_table_current_scope_recursive(
        self,
        node: tree_sitter.Node,
        file_content: str,
        known_classes: Set[str],
        imports: Dict[str, str],
        symbol_table_out: Dict[str, str],
    ):
        if node.type in self.new_scope_node_types:
            return

        def process_declarator(type_node, declarator_node, known_classes, symbol_table_out):
            if type_node and declarator_node:
                type_name_nodes = self._find_nodes_by_type(type_node, 'type_identifier')
                if not type_name_nodes: return
                type_name = self._get_node_text(type_name_nodes[0], file_content)
                
                name_node = declarator_node.child_by_field_name('name')
                if name_node and type_name in known_classes:
                    var_name = self._get_node_text(name_node, file_content)
                    symbol_table_out[var_name] = type_name

        if node.type == 'field_declaration':
            type_node = node.child_by_field_name('type')
            for declarator in self._find_nodes_by_type(node, 'variable_declarator'):
                process_declarator(type_node, declarator, known_classes, symbol_table_out)

        elif node.type == 'formal_parameter':
            type_node = node.child_by_field_name('type')
            name_node = node.child_by_field_name('name')
            if type_node and name_node:
                type_name_nodes = self._find_nodes_by_type(type_node, 'type_identifier')
                if not type_name_nodes: return
                type_name = self._get_node_text(type_name_nodes[0], file_content)

                if type_name in known_classes:
                    param_name = self._get_node_text(name_node, file_content)
                    symbol_table_out[param_name] = type_name

        elif node.type == 'local_variable_declaration':
            type_node = node.child_by_field_name('type')
            for declarator in self._find_nodes_by_type(node, 'variable_declarator'):
                process_declarator(type_node, declarator, known_classes, symbol_table_out)
        
        for child in node.children:
            if child.type not in self.new_scope_node_types:
                self._build_symbol_table_current_scope_recursive(child, file_content, known_classes, imports, symbol_table_out)

    def _look_for_dependencies_scope_recursive(
        self,
        node: tree_sitter.Node,
        file_content: str,
        package: str,
        known_classes: Set[str],
        imports: Dict[str, str],
        dependencies_output: Dict[str, Dict[str, any]],
        parent_symbol_table: Dict[str, str],
        class_name: str,
    ):
        local_symbol_table = parent_symbol_table
        if node.type in self.new_scope_node_types:
            local_symbol_table = parent_symbol_table.copy()
            self._build_symbol_table_current_scope(node, file_content, known_classes, imports, local_symbol_table)
            if self.verbose:
                print(f"Symbol table at {node.type}:")
                for key, value in local_symbol_table.items():
                    print(f"  {key}: {value}")

        # Case 1: Type usage in declarations (fields, params, variables, method returns)
        if node.type in ['field_declaration', 'formal_parameter', 'local_variable_declaration', 'method_declaration']:
            type_node = node.child_by_field_name('type')
            if type_node:
                for type_id_node in self._find_nodes_by_type(type_node, 'type_identifier'):
                    type_name = self._get_node_text(type_id_node, file_content)
                    self._add_dependency(type_name, package, type_id_node.start_point[0] + 1, class_name, known_classes, imports, dependencies_output)

        # Case 2: Object creation `new MyDependency()`
        elif node.type == 'object_creation_expression':
            type_node = node.child_by_field_name('type')
            if type_node:
                for type_id_node in self._find_nodes_by_type(type_node, 'type_identifier'):
                    type_name = self._get_node_text(type_id_node, file_content)
                    self._add_dependency(type_name, package, type_id_node.start_point[0] + 1, class_name, known_classes, imports, dependencies_output)

        # Case 3: Method invocation `dependency.doWork()`
        elif node.type == 'method_invocation':
            obj_node = node.child_by_field_name('object')
            if obj_node:
                var_type = None
                if obj_node.type == 'identifier':
                    var_name = self._get_node_text(obj_node, file_content)
                    var_type = local_symbol_table.get(var_name, var_name)
                elif obj_node.type == 'field_access':
                    if self._get_node_text(obj_node.child_by_field_name('object'), file_content) == 'this':
                        field_name = self._get_node_text(obj_node.child_by_field_name('field'), file_content)
                        var_type = local_symbol_table.get(field_name)
                if var_type:
                    self._add_dependency(var_type, package, obj_node.start_point[0] + 1, class_name, known_classes, imports, dependencies_output)

        # Case 4: Identifier usage (return, assignment)
        elif node.type == 'identifier':
            parent = node.parent
            if parent:
                is_assignment_rhs = parent.type == 'assignment_expression' and parent.child_by_field_name('right') == node
                is_return_value = parent.type == 'return_statement'
                is_field_access = parent.type == 'field_access'

                if is_assignment_rhs or is_return_value or is_field_access:
                    var_name = self._get_node_text(node, file_content)

                    if var_name in known_classes:
                        var_type = var_name
                    else:
                        var_type = local_symbol_table.get(var_name)
                        
                    if var_type:
                        self._add_dependency(var_type, package, node.start_point[0] + 1, class_name, known_classes, imports, dependencies_output)

        for child in node.children:
            self._look_for_dependencies_scope_recursive(
                child, file_content, package, known_classes, imports, dependencies_output, local_symbol_table, class_name
            )

    def extract_methods(
        self,
        file_content: str,
        class_name: str
    ) -> List[Tuple[str, int, int]]:
        """
        Extract public methods from a Java class.
        """
        methods = []
        tree = self._parse_with_tree_sitter(file_content)
        if not tree:
            return methods
            
        class_nodes = self._find_nodes_by_type(tree.root_node, 'class_declaration')
        target_class_node = None
        
        for class_node in class_nodes:
            name_node = class_node.child_by_field_name('name')
            if name_node and self._get_node_text(name_node, file_content) == class_name:
                target_class_node = class_node
                break
                
        if not target_class_node:
            return methods
            
        body_node = target_class_node.child_by_field_name('body')
        if not body_node:
            return methods
            
        method_nodes = self._find_nodes_by_type(body_node, 'method_declaration')
        
        for method_node in method_nodes:
            if self._is_private_method(method_node, file_content):
                continue
                
            name_node = method_node.child_by_field_name('name')
            if not name_node:
                continue
                
            method_name = self._get_node_text(name_node, file_content)
            
            start_line = method_node.start_point[0] + 1
            end_line = method_node.end_point[0] + 1
            
            methods.append((method_name, start_line, end_line))
            
        return methods
        
    def _is_private_method(self, method_node: tree_sitter.Node, file_content: str) -> bool:
        """Check if a method is private."""
        modifiers_node = None
        for child in method_node.children:
            if child.type == 'modifiers':
                modifiers_node = child
                break
                
        if not modifiers_node:
            return False
                
        for modifier in modifiers_node.children:
            modifier_text = self._get_node_text(modifier, file_content)
            if modifier_text == 'private':
                return True
                
        return False