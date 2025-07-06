package com.example.demo.dependency

/**
 * Another dependency class used by KotlinDependency.
 */
class AnotherKotlinDependency(
    private var description: String = "another kotlin dependency"
) {
    /**
     * Get the description.
     */
    fun getDescription(): String {
        return description
    }
    
    /**
     * Set the description.
     */
    fun setDescription(newDescription: String) {
        description = newDescription
    }
}