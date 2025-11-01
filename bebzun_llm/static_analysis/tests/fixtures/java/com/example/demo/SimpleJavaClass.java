package com.example.demo;

import com.example.demo.dependency.JavaDependency;
import com.example.demo.dependency.AnotherJavaDependency;

/**
 * A simple Java class for testing the analyzer.
 */
public class SimpleJavaClass {
    // Constructor parameter with dependency
    private JavaDependency dependency;

    public SimpleJavaClass(JavaDependency dependency) {
        this.dependency = dependency;
    }

    /**
     * Public method that returns a dependency.
     */
    public JavaDependency getDependency() {
        return dependency;
    }
    
    /**
     * Public method that takes a dependency as parameter.
     */
    public void setDependency(JavaDependency dependency) {
        this.dependency = dependency;
    }
    
    /**
     * Public method that uses multiple dependencies.
     */
    public String useMultipleDependencies(
            JavaDependency dep1,
            AnotherJavaDependency dep2
    ) {
        return dep1.getName() + " and " + dep2.getDescription();
    }
    
    /**
     * Private method that should not be included in the output.
     */
    private void privateMethod() {
        System.out.println("This is private");
    }
    
    /**
     * Factory method (static equivalent of Kotlin's companion object).
     */
    public static SimpleJavaClass create() {
        return new SimpleJavaClass(new JavaDependency("default", 0));
    }
}