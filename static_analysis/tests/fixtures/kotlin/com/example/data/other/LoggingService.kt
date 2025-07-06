package com.example.data.other

/**
 * A logging service class used by ConfigurationHelper.
 * A logging service class used by ConfigurationHelper.
 * A logging service class used by ConfigurationHelper.
 */
class LoggingService(
    private var verbose: Boolean = false
) {
    /**
     * Log a message with the current verbosity level.
     */
    fun logMessage(message: String) {
        if (verbose) {
            println("VERBOSE: $message")
        } else {
            println("INFO: $message")
        }
    }

    fun isVerbose(): Boolean {
        return verbose
    }
    

    /**
     * Enable or disable verbose logging.
     */
    fun setVerbose(isVerbose: Boolean) {
        verbose = isVerbose
    }
    
}