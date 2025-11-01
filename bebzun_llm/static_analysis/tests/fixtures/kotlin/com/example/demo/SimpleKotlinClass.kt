package com.example.demo

import com.example.demo.dependency.KotlinDependency
import com.example.demo.dependency.AnotherKotlinDependency

/**
 * A simple Kotlin class for testing the analyzer.
 */
class SimpleKotlinClass(
    // Constructor parameter with dependency
    private var dependency: KotlinDependency
) {
    /**
     * Public method that returns a dependency.
     */
    fun getDependency(): KotlinDependency {
        return dependency
    }
    
    /**
     * Public method that takes a dependency as parameter.
     */
    fun setDependency(dependencyParam: KotlinDependency) {
        this.dependency = dependencyParam
    }
    
    /**
     * Public method that uses multiple dependencies.
     */
    fun useMultipleDependencies(
        dep1: KotlinDependency,
        dep2: AnotherKotlinDependency
    ): String {
        return "${dep1.getName()} and ${dep2.getDescription()}"
    }
    
    /**
     * Private method that should not be included in the output.
     */
    private fun privateMethod() {
        println("This is private")
    }
    
    /**
     * Companion object with a factory method.
     */
    companion object {
        fun create(): SimpleKotlinClass {
            return SimpleKotlinClass(KotlinDependency("default", 0))
        }
    }
}