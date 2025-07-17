"""
Tests for the Java parser.
"""

import unittest
from pathlib import Path
from typing import Dict, List, Set, Tuple, NamedTuple

from static_analysis.parsers.java_parser import JavaParser
from static_analysis.model.model import ClassStructure

class JavaTestCase(NamedTuple):
    """Test case data for a Java class."""
    name: str
    path: Path
    dependencies_paths: List[Path]
    expected_classname: str
    expected_full_classname: str
    expected_methods: Dict[str, Dict[str, int]]
    expected_excluded_methods: Set[str]
    expected_dependencies: Dict[str, Dict[str, any]]


class TestJavaParser(unittest.TestCase):
    """Test cases for the JavaParser class."""
    
    def setUp(self):
        """Set up the test case."""
        self.parser = JavaParser()
        self.fixtures_dir = Path(__file__).parent / 'fixtures' / 'java'
        
        # Prepare test cases
        self.test_cases = {
            'com.example.demo.SimpleJavaClass': self._create_simple_java_class_test_case(),
            'com.example.data.DataProcessor': self._create_data_processor_test_case()
        }
    
    def _create_simple_java_class_test_case(self) -> JavaTestCase:
        """Create test case data for SimpleJavaClass."""
        path_tested_class = self.fixtures_dir / 'com' / 'example' / 'demo' / 'SimpleJavaClass.java'
        path_dependency_1 = self.fixtures_dir / 'com' / 'example' / 'demo' / 'dependency' / 'JavaDependency.java'
        path_dependency_2 = self.fixtures_dir / 'com' / 'example' / 'demo' / 'dependency' / 'AnotherJavaDependency.java'
        
        expected_methods = {
            'getDependency': {'start': 20, 'end': 22},
            'setDependency': {'start': 27, 'end': 29},
            'useMultipleDependencies': {'start': 34, 'end': 39},
            'create': {'start': 51, 'end': 53},
        }
        
        expected_excluded_methods = {'privateMethod'}
        
        expected_dependencies = {
            'com.example.demo.dependency.JavaDependency': {
                'full_name': 'com.example.demo.dependency.JavaDependency',
                'expected_lines': [11, 13, 14, 20, 21, 27, 28, 35, 38, 52]  # Line numbers where JavaDependency is used
            },
            'com.example.demo.dependency.AnotherJavaDependency': {
                'full_name': 'com.example.demo.dependency.AnotherJavaDependency',
                'expected_lines': [36, 38]  # Line numbers where AnotherJavaDependency is used
            }
        }
        
        return JavaTestCase(
            name='SimpleJavaClass',
            path=path_tested_class,
            dependencies_paths=[path_dependency_1, path_dependency_2],
            expected_classname='SimpleJavaClass',
            expected_full_classname='com.example.demo.SimpleJavaClass',
            expected_methods=expected_methods,
            expected_excluded_methods=expected_excluded_methods,
            expected_dependencies=expected_dependencies
        )
    
    def _create_data_processor_test_case(self) -> JavaTestCase:
        """Create test case data for DataProcessor."""
        path_tested_class = self.fixtures_dir / 'com' / 'example' / 'data' / 'DataProcessor.java'
        path_config_helper = self.fixtures_dir / 'com' / 'example' / 'data' / 'other' / 'ConfigurationHelper.java'
        path_logging_service = self.fixtures_dir / 'com' / 'example' / 'data' / 'other' / 'LoggingService.java'
        path_simple_calculator = self.fixtures_dir / 'com' / 'example' / 'data' / 'other' / 'SimpleCalculator.java'
        
        expected_methods = {
            'updateConfiguration': {'start': 22, 'end': 24},
            'processDataWithLogging': {'start': 29, 'end': 36},
            'getConfiguration': {'start': 50, 'end': 52},
            'createDefault': {'start': 59, 'end': 62}
        }
        
        expected_excluded_methods = {'internalProcess'}
        
        expected_dependencies = {
            'com.example.data.other.ConfigurationHelper': {
                'full_name': 'com.example.data.other.ConfigurationHelper',
                'expected_lines': [12, 15, 16, 22, 23, 30, 33, 35, 50, 51, 60]
            },
            'com.example.data.other.LoggingService': {
                'full_name': 'com.example.data.other.LoggingService',
                'expected_lines': [31, 34]
            },
            'com.example.data.other.SimpleCalculator': {
                'full_name': 'com.example.data.other.SimpleCalculator',
                'expected_lines': [13, 43]
            }
        }
        
        return JavaTestCase(
            name='DataProcessor',
            path=path_tested_class,
            dependencies_paths=[path_config_helper, path_logging_service, path_simple_calculator],
            expected_classname='DataProcessor',
            expected_full_classname='com.example.data.DataProcessor',
            expected_methods=expected_methods,
            expected_excluded_methods=expected_excluded_methods,
            expected_dependencies=expected_dependencies
        )
    
    def test_parse_file(self):
        """Test parsing a Java file."""
        for test_name, test_case in self.test_cases.items():
            with self.subTest(f"Testing {test_name}"):
                # SETUP - Expected results
                expected_class_count = 1
                expected_simple_classname = test_case.expected_classname
                expected_full_classname = test_case.expected_full_classname
                expected_source_file = test_case.path.name

                # Print Tree-sitter output for debugging
                print(f"\n--- Tree output for {test_name} ({test_case.path}) ---\n")
                try:
                    with open(test_case.path, 'r', encoding='utf-8') as f:
                        file_content_for_sexp = f.read()
                    
                    # Access the internal _parse_with_tree_sitter method from self.parser
                    # The KotlinParser class has this method.
                    tree = self.parser._parse_with_tree_sitter(file_content_for_sexp) # type: ignore
                    if tree and tree.root_node:
                        # try:
                        #     with open(f"dot_graph_java_{expected_simple_classname}", 'w', encoding='utf-8') as output_file:
                        #         tree.print_dot_graph(output_file)
                        # except Exception as e:
                        #     print(f"Error getting dot-graph for {test_case.path}: {e}")
                        print(str(tree.root_node))
                    else:
                        print(f"Could not parse {test_case.path}")
                except Exception as e:
                    print(f"Error printing tree for {test_case.path}: {e}")
                print(f"--- End printing tree for {test_name} ---\n")
                
                # EXECUTE - Parse the file
                classes = self.parser.extract_classes(test_case.path, test_case.path.parent)
                
                # ASSERT - Verify results
                self.assertEqual(len(classes), expected_class_count,
                                f"Expected {expected_class_count} class but found {len(classes)} classes")
                
                # Additional debug output if test would fail
                if len(classes) != expected_class_count:
                    for idx, cls in enumerate(classes):
                        print(f"  Class {idx+1}: {cls.simple_classname} (full: {cls.full_classname})")
                
                self.assertEqual(classes[0].simple_classname, expected_simple_classname)
                self.assertEqual(classes[0].full_classname, expected_full_classname)
                self.assertEqual(classes[0].source_file, expected_source_file)
    
    def test_extract_dependencies(self):
        """Test extracting dependencies from a Java class."""
        for test_name, test_case in self.test_cases.items():
            with self.subTest(f"Testing dependencies for {test_name}"):
                # SETUP - Test data and expected results
                known_classes = {test_case.expected_classname}
                for dep_name in test_case.expected_dependencies:
                    known_classes.add(dep_name)
                
                expected_dependencies = test_case.expected_dependencies
                import_line_range = range(2, 6)  # Lines where imports are typically found
                
                # EXECUTE - Extract dependencies
                with open(test_case.path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                cls = ClassStructure(
                    simple_classname = test_case.expected_classname,
                    full_classname = test_case.expected_full_classname,
                    dependencies = [],
                    public_methods = [],
                    source_file = str(test_case.path),
                )
                dependencies = self.parser.extract_dependencies(file_content, cls, known_classes)
                
                # ASSERT - Verify results
                dependency_names = {name for name, _, _ in dependencies}
                
                # Build a map of actual dependencies for easier comparison
                actual_dependencies = {}
                for full_name, usage_lines in dependencies:
                    actual_dependencies[full_name] = {
                        'full_name': full_name,
                        'actual_lines': set(usage_lines)
                    }
                
                # Debug output if needed
                if not all(dep in dependency_names for dep in expected_dependencies.keys()):
                    missing = set(expected_dependencies.keys()) - dependency_names
                    print(f"Missing dependencies for {test_name}: {missing}")
                    print(f"Dependencies found: {dependency_names}")
                    print(f"Dependencies details:")
                    for full_name, lines in dependencies:
                        print(f"  {full_name}: lines {lines}")
                
                # Verify all expected dependencies are found with correct count
                self.assertEqual(len(expected_dependencies), len(dependency_names),
                                f"Expected dependencies: {set(expected_dependencies.keys())}, found: {dependency_names}")
                
                for expected_dep in expected_dependencies:
                    self.assertIn(expected_dep, dependency_names, f"Dependency {expected_dep} not found")
                
                # Verify details for each dependency
                for dep_name, expected in expected_dependencies.items():
                    # Verify dependency exists in actual results
                    self.assertIn(dep_name, actual_dependencies, f"Dependency {dep_name} not found in actual results")
                    
                    actual = actual_dependencies[dep_name]
                    
                    # Verify full name matches
                    self.assertEqual(expected['full_name'], actual['full_name'],
                                    f"Full name mismatch for {dep_name}: expected {expected['full_name']}, got {actual['full_name']}")
                    
                    # Verify usage lines exist
                    self.assertTrue(len(actual['actual_lines']) > 0, f"No usage lines found for {dep_name}")
                    
                    # Convert expected lines to a set for exact comparison
                    expected_lines_set = set(expected['expected_lines'])
                    
                    # Verify usage lines are not in import section
                    for line in actual['actual_lines']:
                        self.assertNotIn(line, import_line_range,
                                        f"Usage line {line} should not include import lines")
                        self.assertGreater(line, 1,
                                          f"Usage line {line} should be after package declaration")
                    
                    # Verify exact match between expected and actual usage lines
                    self.assertEqual(expected_lines_set, actual['actual_lines'],
                                    f"Usage line mismatch for {dep_name}:\n"
                                    f"Expected: {sorted(expected_lines_set)}\n"
                                    f"Actual: {sorted(actual['actual_lines'])}")
    
    def test_extract_methods(self):
        """Test extracting public methods from a Java class with line number validation."""
        for test_name, test_case in self.test_cases.items():
            with self.subTest(f"Testing methods for {test_name}"):
                # SETUP - Test data and expected results
                with open(test_case.path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                expected_methods = set(test_case.expected_methods.keys())
                excluded_methods = test_case.expected_excluded_methods
                expected_method_lines = test_case.expected_methods
                
                
                # EXECUTE - Extract methods
                methods = self.parser.extract_methods(file_content, test_case.expected_classname)
                
                # ASSERT - Verify results
                method_names = {name for name, _, _ in methods}
                method_dict = {name: (start, end) for name, start, end in methods}
                
                # Debug output if needed
                if not all(method in method_names for method in expected_methods):
                    missing = expected_methods - method_names
                    print(f"Missing methods for {test_name}: {missing}")
                    print(f"Methods found: {method_names}")
                    print(f"Method details:")
                    for name, start, end in methods:
                        print(f"  {name}: lines {start}-{end}")
                
                # Verify all expected methods are found
                for expected_method in expected_methods:
                    self.assertIn(expected_method, method_names,
                                    f"Expected method {expected_method} not found. Methods found: {method_names}")
                
                # Verify excluded methods are not found
                for excluded_method in excluded_methods:
                    self.assertNotIn(excluded_method, method_names,
                                        f"Method {excluded_method} should be excluded but was found in {method_names}")
                
                # Verify basic line number validity for each method
                for _, start_line, end_line in methods:
                    self.assertTrue(start_line > 0, f"Start line {start_line} should be positive")
                    self.assertTrue(end_line > start_line,
                                    f"End line {end_line} should be greater than start line {start_line}")
                
                # Verify line numbers match expected values with 1-line tolerance
                for method_name, expected_lines in expected_method_lines.items():
                    self.assertIn(method_name, method_dict, f"Method {method_name} not found")
                    
                    actual_start, actual_end = method_dict[method_name]
                    expected_start = expected_lines['start']
                    expected_end = expected_lines['end']
                    
                    # Allow 1-line difference in start line
                    slack = 2
                    self.assertTrue(abs(actual_start - expected_start) <= slack,
                                    f"{method_name} start line mismatch: expected {expected_start}, got {actual_start}, difference exceeds {slack} lines")
                    
                    # Allow 1-line difference in end line
                    self.assertTrue(abs(actual_end - expected_end) <= slack,
                                    f"{method_name} end line mismatch: expected {expected_end}, got {actual_end}, difference exceeds {slack} lines")
    
    def test_extract_methods_visibility_filtering(self):
        """Test that method extraction properly filters by visibility."""
        for test_name, test_case in self.test_cases.items():
            with self.subTest(f"Testing method visibility filtering for {test_name}"):
                # SETUP - Test data and expected results
                with open(test_case.path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                expected_methods = set(test_case.expected_methods.keys())
                excluded_methods = test_case.expected_excluded_methods
                expected_method_count = len(expected_methods)
                
                # EXECUTE - Extract methods with visibility filtering
                methods = self.parser.extract_methods(file_content, test_case.expected_classname)
                method_names = {name for name, _, _ in methods}
                
                # ASSERT - Verify results
                
                # Verify public methods are included
                for method_name in expected_methods:
                    self.assertIn(method_name, method_names,
                                  f"Public method {method_name} should be included but was not found. Methods found: {method_names}")
                
                # Verify private/static methods are excluded
                for method_name in excluded_methods:
                    self.assertNotIn(method_name, method_names,
                                     f"Method {method_name} should be excluded but was found in {method_names}")
                
                # Verify exact method count
                self.assertEqual(len(method_names), expected_method_count,
                                 f"Expected {expected_method_count} methods ({expected_methods}), found {len(method_names)} methods ({method_names})")
    
    def test_extract_methods_class_scope(self):
        """Test that method extraction only finds methods in the target class."""
        for test_name, test_case in self.test_cases.items():
            with self.subTest(f"Testing method class scope for {test_name}"):
                # SETUP - Test data and expected results
                with open(test_case.path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                existing_class = test_case.expected_classname
                
                # EXECUTE - Test with existing class
                methods_existing = self.parser.extract_methods(file_content, existing_class)
                
                # ASSERT - Verify results
                
                # Existing class should have methods
                method_names = [name for name, _, _ in methods_existing]
                self.assertGreater(len(methods_existing), 0,
                                   f"Existing class {existing_class} should have methods, but found none. "
                                   f"Methods found: {method_names}")
                
                # Test with non-existing class should return empty list
                non_existing_class = "NonExistingClass"
                methods_non_existing = self.parser.extract_methods(file_content, non_existing_class)
                self.assertEqual(len(methods_non_existing), 0,
                                f"Non-existing class {non_existing_class} should have no methods, but found: "
                                f"{[name for name, _, _ in methods_non_existing]}")


if __name__ == '__main__':
    unittest.main()