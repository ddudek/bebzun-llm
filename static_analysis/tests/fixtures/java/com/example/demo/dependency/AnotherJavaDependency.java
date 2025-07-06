package com.example.demo.dependency;

/**
 * Another dependency class used by JavaDependency.
 */
public class AnotherJavaDependency {
    private String description = "another java dependency";

    public AnotherJavaDependency() {
    }

    public AnotherJavaDependency(String description) {
        this.description = description;
    }

    /**
     * Get the description.
     */
    public String getDescription() {
        return description;
    }
    
    /**
     * Set the description.
     */
    public void setDescription(String newDescription) {
        this.description = newDescription;
    }
}