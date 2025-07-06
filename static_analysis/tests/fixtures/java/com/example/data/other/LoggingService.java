package com.example.data.other;

/**
 * A logging service class used by ConfigurationHelper.
 * A logging service class used by ConfigurationHelper.
 * A logging service class used by ConfigurationHelper.
 */
public class LoggingService {
    private boolean verbose = false;

    public LoggingService() {
    }

    public LoggingService(boolean verbose) {
        this.verbose = verbose;
    }

    /**
     * Log a message with the current verbosity level.
     */
    public void logMessage(String message) {
        if (verbose) {
            System.out.println("VERBOSE: " + message);
        } else {
            System.out.println("INFO: " + message);
        }
    }

    public boolean isVerbose() {
        return verbose;
    }

    /**
     * Enable or disable verbose logging.
     */
    public void setVerbose(boolean isVerbose) {
        this.verbose = isVerbose;
    }
}