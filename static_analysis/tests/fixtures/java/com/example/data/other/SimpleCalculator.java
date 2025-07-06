package com.example.data.other;

/**
 * A simple calculator class used by DataProcessor.
 */
public class SimpleCalculator {
    private int precision = 2;

    public SimpleCalculator() {
    }

    public SimpleCalculator(int precision) {
        this.precision = precision;
    }

    /**
     * Add two numbers.
     */
    public double add(double a, double b) {
        return a + b;
    }

    /**
     * Subtract two numbers.
     */
    public double subtract(double a, double b) {
        return a - b;
    }

    /**
     * Multiply two numbers.
     */
    public double multiply(double a, double b) {
        return a * b;
    }

    /**
     * Divide two numbers.
     */
    public double divide(double a, double b) {
        if (b == 0.0) {
            throw new IllegalArgumentException("Cannot divide by zero");
        }
        return a / b;
    }

    /**
     * Set the calculation precision.
     */
    public void setPrecision(int newPrecision) {
        if (newPrecision <= 0) {
            throw new IllegalArgumentException("Precision must be positive");
        }
        this.precision = newPrecision;
    }

    /**
     * Get the current calculation precision.
     */
    public int getPrecision() {
        return precision;
    }
}