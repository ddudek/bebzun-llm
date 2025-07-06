package com.example.demo.dependency;

import java.util.Objects;

/**
 * A dependency class used by SimpleJavaClass.
 */
public class JavaDependency {
    private String name;
    private int value;

    public JavaDependency(String name, int value) {
        this.name = name;
        this.value = value;
    }

    /**
     * Get the name.
     */
    public String getName() {
        return name;
    }
    
    /**
     * Set the name.
     */
    public void setName(String newName) {
        this.name = newName;
    }
    
    /**
     * Get the value.
     */
    public int getValue() {
        return value;
    }
    
    /**
     * Set the value.
     */
    public void setValue(int newValue) {
        this.value = newValue;
    }
    
    /**
     * A method that uses another dependency.
     */
    public void useAnotherDependency(AnotherJavaDependency dependency) {
        System.out.println("Using " + dependency.getDescription());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        JavaDependency that = (JavaDependency) o;
        return value == that.value && Objects.equals(name, that.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, value);
    }

    @Override
    public String toString() {
        return "JavaDependency{" +
                "name='" + name + '\'' +
                ", value=" + value +
                '}';
    }
}