package com.example.data.other

/**
 * A simple calculator class used by DataProcessor.
 */
class SimpleCalculator(
    private var precision: Int = 2
) {
    /**
     * Add two numbers.
     */
    fun add(a: Double, b: Double): Double {
        return a + b
    }

    /**
     * Subtract two numbers.
     */
    fun subtract(a: Double, b: Double): Double {
        return a - b
    }

    /**
     * Multiply two numbers.
     */
    fun multiply(a: Double, b: Double): Double {
        return a * b
    }

    /**
     * Divide two numbers.
     */
    fun divide(a: Double, b: Double): Double {
        require(b != 0.0) { "Cannot divide by zero" }
        return a / b
    }

    /**
     * Set the calculation precision.
     */
    fun setPrecision(newPrecision: Int) {
        require(newPrecision > 0) { "Precision must be positive" }
        precision = newPrecision
    }

    /**
     * Get the current calculation precision.
     */
    fun getPrecision(): Int {
        return precision
    }
}