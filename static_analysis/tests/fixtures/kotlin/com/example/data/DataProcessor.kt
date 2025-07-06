package com.example.data

import com.example.data.other.ConfigurationHelper
import com.example.data.other.LoggingService
import com.example.data.other.SimpleCalculator

/**
 * A data processing class for testing the analyzer.
 */
class DataProcessor(
    // Constructor parameter with dependency
    private var config: ConfigurationHelper
) {
    private val calculator: SimpleCalculator = SimpleCalculator()
    
    /**
     * Public method that updates the configuration.
     */
    fun updateConfiguration(
        newConfig: ConfigurationHelper
    ) {
        this.config = newConfig
    }
    
    /**
     * Public method that processes data using multiple dependencies.
     */
    fun processDataWithLogging(
        configLocal: ConfigurationHelper,
        logger: LoggingService
    ): Boolean {
        var one = configLocal.isEnabled()
        var two = logger.isVerbose()
        return one && two
    }
    
    /**
     * Private method that should not be included in the output.
     */
    private fun internalProcess() {
        println("Processing data internally")
        val result = calculator.add(10.0, 5.0)
        println("Calculation result: $result")
    }

    /**
     * Public method that returns the configuration.
     */
    fun getConfiguration(): ConfigurationHelper {
        return config
    }
    
    /**
     * Companion object 
     * with a factory method.
     */
    companion object {
        fun createDefault(): DataProcessor {
            var helper = ConfigurationHelper("default", true)
            return DataProcessor(
                helper
            )
        }
    }
}

/**
 * A simple data class in the same file as DataProcessor.
 */
data class ProcessingMetadata(
    val timestamp: Long,
    val source: String
)