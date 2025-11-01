package com.example.data;

import com.example.data.other.ConfigurationHelper;
import com.example.data.other.LoggingService;
import com.example.data.other.SimpleCalculator;

/**
 * A data processing class for testing the analyzer.
 */
public class DataProcessor {
    // Constructor parameter with dependency
    private ConfigurationHelper config;
    private final SimpleCalculator calculator = new SimpleCalculator();
    
    public DataProcessor(ConfigurationHelper config) {
        this.config = config;
    }
    
    /**
     * Public method that updates the configuration.
     */
    public void updateConfiguration(ConfigurationHelper newConfig) {
        this.config = newConfig;
    }
    
    /**
     * Public method that processes data using multiple dependencies.
     */
    public boolean processDataWithLogging(
            ConfigurationHelper config,
            LoggingService logger
    ) {
        boolean one = config.isEnabled();
        boolean two = logger.isVerbose();
        return one && two && ConfigurationHelper.FLAG_ENABLED;
    }
    
    /**
     * Private method that should not be included in the output.
     */
    private void internalProcess() {
        System.out.println("Processing data internally");
        double result = calculator.add(10.0, 5.0);
        System.out.println("Calculation result: " + result);
    }

    /**
     * Public method that returns the configuration.
     */
    public ConfigurationHelper getConfiguration() {
        return config;
    }
    

    
    /**
     * Static factory method (equivalent to Kotlin's companion object).
     */
    public static DataProcessor createDefault() {
        ConfigurationHelper helper = new ConfigurationHelper("default", true);
        return new DataProcessor(helper);
    }
}