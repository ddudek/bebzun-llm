"""
Tests for the Kotlin parser.
"""

import unittest
from pathlib import Path
from typing import Dict, List, Set, Tuple, NamedTuple

from static_analysis.parsers.kotlin_parser import KotlinParser
from static_analysis.model.model import ClassStructure

class KotlinTestCase(NamedTuple):
    """Test case data for a Kotlin class."""
    name: str
    path: Path
    dependencies_paths: List[Path]
    expected_class_count: str
    expected_classname: str
    expected_full_classname: str
    expected_methods: Dict[str, Dict[str, int]]
    expected_excluded_methods: Set[str]
    expected_dependencies: Dict[str, Dict[str, any]]

class TestKotlinParser(unittest.TestCase):
    """Test cases for the KotlinParser class."""
    
    def setUp(self):
        """Set up the test case."""
        self.parser = KotlinParser()
        self.fixtures_dir = Path(__file__).parent / 'fixtures' / 'kotlin'
        
        # Prepare test cases
        self.test_cases = {
            'com.example.demo.SimpleKotlinClass': self._create_simple_kotlin_class_test_case(),
            'com.example.data.DataProcessor': self._create_data_processor_test_case(),
            'com.example.demo.ProcessingMetadata': self._create_processing_metadata_test_case()
        }
    
    def _create_simple_kotlin_class_test_case(self) -> KotlinTestCase:
        """Create test case data for SimpleKotlinClass."""
        path_tested_class = self.fixtures_dir / 'com' / 'example' / 'demo' / 'SimpleKotlinClass.kt'
        path_dependency_1 = self.fixtures_dir / 'com' / 'example' / 'demo' / 'dependency' / 'KotlinDependency.kt'
        path_dependency_2 = self.fixtures_dir / 'com' / 'example' / 'demo' / 'dependency' / 'AnotherKotlinDependency.kt'
        
        expected_methods = {
            'getDependency': {'start': 16, 'end': 18},
            'setDependency': {'start': 23, 'end': 25},
            'useMultipleDependencies': {'start': 30, 'end': 35}
        }
        
        expected_excluded_methods = {'privateMethod', 'create'}
        
        expected_dependencies = {
            'com.example.demo.dependency.KotlinDependency': {
                'full_name': 'com.example.demo.dependency.KotlinDependency',
                'expected_lines': [11, 16, 17, 23, 24, 31, 34, 49]  # Line numbers where KotlinDependency is used
            },
            'com.example.demo.dependency.AnotherKotlinDependency': {
                'full_name': 'com.example.demo.dependency.AnotherKotlinDependency',
                'expected_lines': [32, 34]  # Line numbers where AnotherKotlinDependency is used
            }
        }
        
        return KotlinTestCase(
            name='SimpleKotlinClass',
            path=path_tested_class,
            dependencies_paths=[path_dependency_1, path_dependency_2],
            expected_class_count=1,
            expected_classname='SimpleKotlinClass',
            expected_full_classname='com.example.demo.SimpleKotlinClass',
            expected_methods=expected_methods,
            expected_excluded_methods=expected_excluded_methods,
            expected_dependencies=expected_dependencies
        )
    
    def _create_data_processor_test_case(self) -> KotlinTestCase:
        """Create test case data for DataProcessor."""
        path_tested_class = self.fixtures_dir / 'com' / 'example' / 'data' / 'DataProcessor.kt'
        path_config_helper = self.fixtures_dir / 'com' / 'example' / 'data' / 'other' / 'ConfigurationHelper.kt'
        path_logging_service = self.fixtures_dir / 'com' / 'example' / 'data' / 'other' / 'LoggingService.kt'
        path_simple_calculator = self.fixtures_dir / 'com' / 'example' / 'data' / 'other' / 'SimpleCalculator.kt'
        
        expected_methods = {
            'updateConfiguration': {'start': 19, 'end': 23},
            'processDataWithLogging': {'start': 28, 'end': 35},
            'getConfiguration': {'start': 49, 'end': 51}
        }
        
        expected_excluded_methods = {'internalProcess', 'createDefault'}
        
        expected_dependencies = {
            'com.example.data.other.ConfigurationHelper': {
                'full_name': 'com.example.data.other.ConfigurationHelper',
                'expected_lines': [12, 20, 22, 29, 32, 34, 49, 50, 59, 61]
            },
            'com.example.data.other.LoggingService': {
                'full_name': 'com.example.data.other.LoggingService',
                'expected_lines': [30, 33]
            },
            'com.example.data.other.SimpleCalculator': {
                'full_name': 'com.example.data.other.SimpleCalculator',
                'expected_lines': [14, 42]
            }
        }
        
        return KotlinTestCase(
            name='DataProcessor',
            path=path_tested_class,
            dependencies_paths=[path_config_helper, path_logging_service, path_simple_calculator],
            expected_class_count=2,
            expected_classname='DataProcessor',
            expected_full_classname='com.example.data.DataProcessor',
            expected_methods=expected_methods,
            expected_excluded_methods=expected_excluded_methods,
            expected_dependencies=expected_dependencies
        )
    
    def _create_processing_metadata_test_case(self) -> KotlinTestCase:
        """Create test case data for ProcessingMetadata."""
        path_tested_class = self.fixtures_dir / 'com' / 'example' / 'data' / 'DataProcessor.kt'
        
        # Data class has no methods to test
        expected_methods = {}
        expected_excluded_methods = set()
        expected_dependencies = {}
        
        return KotlinTestCase(
            name='com.example.data.ProcessingMetadata',
            path=path_tested_class,
            dependencies_paths=[],
            expected_class_count=2,
            expected_classname='ProcessingMetadata',
            expected_full_classname='com.example.data.ProcessingMetadata',
            expected_methods=expected_methods,
            expected_excluded_methods=expected_excluded_methods,
            expected_dependencies=expected_dependencies
        )
    
    def test_parse_file(self):
        """Test parsing a Kotlin file."""
        for test_name, test_case in self.test_cases.items():
            with self.subTest(f"Testing {test_name}"):
                # SETUP - Expected results
                # For DataProcessor.kt file we expect 2 classes
                expected_class_count = test_case.expected_class_count
                expected_simple_classname = test_case.expected_classname
                expected_full_classname = test_case.expected_full_classname
                expected_source_file = test_case.path.name
                
                # Print Tree-sitter S-expression for debugging
                print(f"\n--- Tree output for {test_name} ({test_case.path}) ---")
                try:
                    with open(test_case.path, 'r', encoding='utf-8') as f:
                        file_content_for_sexp = f.read()
                    
                    # Access the internal _parse_with_tree_sitter method from self.parser
                    # The KotlinParser class has this method.
                    tree = self.parser._parse_with_tree_sitter(file_content_for_sexp) # type: ignore
                    if tree and tree.root_node:
                        # try:
                        #     with open(f"dot_graph_{expected_simple_classname}", 'w', encoding='utf-8') as output_file:
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
                
                # Find the class with matching simple_classname
                found_class = None
                for cls in classes:
                    if cls.simple_classname == expected_simple_classname:
                        found_class = cls
                        break
                
                self.assertIsNotNone(found_class, f"Class {expected_simple_classname} not found in parsed classes")
                
                # Verify all classes are expected
                if test_name in ['DataProcessor', 'ProcessingMetadata']:
                    expected_classes = {'DataProcessor', 'ProcessingMetadata'}
                    actual_classes = {cls.simple_classname for cls in classes}
                    self.assertEqual(expected_classes, actual_classes,
                                   f"Found unexpected classes. Expected: {expected_classes}, Got: {actual_classes}")
                
                # Verify class properties
                self.assertEqual(found_class.simple_classname, expected_simple_classname)
                self.assertEqual(found_class.full_classname, expected_full_classname)
                self.assertEqual(found_class.source_file, expected_source_file)
    
    def test_extract_dependencies(self):
        """Test extracting dependencies from a Kotlin class."""
        # Enable debug mode to help diagnose indirect dependency detection issues
        original_debug = getattr(self.parser, 'debug', False)
        self.parser.debug = True
        
        for test_name, test_case in self.test_cases.items():
            with self.subTest(f"Testing dependencies for {test_name}"):
                print(f"\n{'='*80}\nTESTING DEPENDENCIES FOR: {test_name}\n{'='*80}")
                
                # SETUP - Test data and expected results
                known_classes = {test_case.expected_full_classname}
                for dep_name in test_case.expected_dependencies:
                    known_classes.add(dep_name)
                
                expected_dependencies = test_case.expected_dependencies
                import_line_range = range(2, 6)  # Lines where imports are typically found
                
                print(f"\nExpected dependencies for {test_name}:")
                for dep_name, dep_info in expected_dependencies.items():
                    print(f"  {dep_name} ({dep_info['full_name']}): lines {sorted(dep_info['expected_lines'])}")
                
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
                dependency_names = {name for name, _ in dependencies}
                
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
                    for name, full_name, lines in dependencies:
                        print(f"  {name} ({full_name}): lines {lines}")
                
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
                                    f"\nUsage line mismatch for {dep_name}:\n"
                                    f"Expected: {sorted(expected_lines_set)}\n"
                                    f"Actual: {sorted(actual['actual_lines'])}")
        
        # Restore original debug setting
        self.parser.debug = original_debug
    
    def test_extract_methods_tree_sitter_specific(self):
        """Test tree-sitter specific method extraction functionality."""
        for test_name, test_case in self.test_cases.items():
            with self.subTest(f"Testing method line numbers for {test_name}"):
                # SETUP - Test data and expected results
                with open(test_case.path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                expected_method_lines = test_case.expected_methods
                
                # Enable debug to see which method is being used
                original_debug = getattr(self.parser, 'debug', False)
                self.parser.debug = True
                
                try:
                    # EXECUTE - Extract methods with tree-sitter
                    methods = self.parser.extract_methods(file_content, test_case.expected_classname)
                    
                    # ASSERT - Verify results
                    method_dict = {name: (start, end) for name, start, end in methods}
                    
                    # Verify precise line numbers for each method
                    for method_name, expected_lines in expected_method_lines.items():
                        self.assertIn(method_name, method_dict, f"Method {method_name} not found")
                        
                        actual_start, actual_end = method_dict[method_name]
                        self.assertEqual(actual_start, expected_lines['start'],
                                        f"{method_name} start line mismatch: expected {expected_lines['start']}, got {actual_start}")
                        self.assertEqual(actual_end, expected_lines['end'],
                                        f"{method_name} end line mismatch: expected {expected_lines['end']}, got {actual_end}")
                finally:
                    self.parser.debug = original_debug
    
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
                
                # Verify private/companion methods are excluded
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
                # todo: check if methods match expected methods.
                
                # Test with non-existing class should return empty list
                non_existing_class = "NonExistingClass"
                methods_non_existing = self.parser.extract_methods(file_content, non_existing_class)
                self.assertEqual(len(methods_non_existing), 0,
                                f"Non-existing class {non_existing_class} should have no methods, but found: "
                                f"{[name for name, _, _ in methods_non_existing]}")

if __name__ == '__main__':
    unittest.main()