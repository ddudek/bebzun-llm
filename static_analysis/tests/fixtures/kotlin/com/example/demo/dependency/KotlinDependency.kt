package com.example.demo.dependency

import com.example.demo.dependency.AnotherKotlinDependency

/**
 * A dependency class used by SimpleKotlinClass.
 */
data class KotlinDependency(
    private var name: String,
    private var value: Int
) {
    /**
     * Get the name.
     */
    fun getName(): String {
        return name
    }
    
    /**
     * Set the name.
     */
    fun setName(newName: String) {
        name = newName
    }
    
    /**
     * Get the value.
     */
    fun getValue(): Int {
        return value
    }
    
    /**
     * Set the value.
     */
    fun setValue(newValue: Int) {
        value = newValue
    }
    
    /**
     * A method that uses another dependency.
     */
    fun useAnotherDependency(dependency: AnotherKotlinDependency) {
        println("Using ${dependency.getDescription()}")
    }
}