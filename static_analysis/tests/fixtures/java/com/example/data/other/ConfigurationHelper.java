package com.example.data.other;

import java.util.Objects;

/**
 * A configuration helper class used by DataProcessor.
 */
public class ConfigurationHelper {
    private String name;
    private boolean enabled;

    public ConfigurationHelper(String name, boolean enabled) {
        this.name = name;
        this.enabled = enabled;
    }

    /**
     * A method that uses the logging service.
     */
    public void configureWithLogging(LoggingService logger) {
        System.out.println("Configured with logging level: " + logger.isVerbose());
    }

    public String getName() {
        return name;
    }
    
    /**
     * Set the configuration name.
     */
    public void setName(String newName) {
        this.name = newName;
    }
    
    /**
     * Enable or disable the configuration.
     */
    public void setEnabled(boolean isEnabled) {
        this.enabled = isEnabled;
    }
    
    /**
     * Check if the configuration is enabled.
     */
    public boolean isEnabled() {
        return enabled;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ConfigurationHelper that = (ConfigurationHelper) o;
        return enabled == that.enabled && Objects.equals(name, that.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, enabled);
    }

    @Override
    public String toString() {
        return "ConfigurationHelper{" +
                "name='" + name + '\'' +
                ", enabled=" + enabled +
                '}';
    }

    public final static boolean FLAG_ENABLED = false;
}