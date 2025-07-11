package com.example.data.other

import com.example.data.other.LoggingService

/**
 * A configuration helper class used by DataProcessor.
 */
data class ConfigurationHelper(
    private var name: String,
    private var enabled: Boolean
) {
    /**
     * A method that uses the logging service.
     */
    fun configureWithLogging(logger: LoggingService) {
        println("Configured with logging level: ${logger.isVerbose()}")
    }

    fun getName(): String {
        return name
    }
    
    /**
     * Set the configuration name.
     */
    fun setName(newName: String) {
        name = newName
    }
    
    /**
     * Enable or disable the configuration.
     */
    fun setEnabled(isEnabled: Boolean) {
        enabled = isEnabled
    }
    
    /**
     * Check if the configuration is enabled.
     */
    fun isEnabled(): Boolean {
        return enabled
    }

    
}