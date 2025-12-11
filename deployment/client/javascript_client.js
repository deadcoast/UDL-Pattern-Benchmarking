/**
 * JavaScript client for UDL Rating Framework API.
 * 
 * This module provides a convenient JavaScript interface for interacting with
 * the UDL Rating Framework REST API.
 */

class UDLRatingClient {
    /**
     * Initialize the client.
     * 
     * @param {string} baseUrl - Base URL of the API
     * @param {string|null} apiToken - API authentication token
     * @param {number} timeout - Request timeout in milliseconds
     */
    constructor(baseUrl = 'http://localhost:8000', apiToken = null, timeout = 30000) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.apiToken = apiToken;
        this.timeout = timeout;
        
        // Default headers
        this.headers = {
            'Content-Type': 'application/json',
        };
        
        if (apiToken) {
            this.headers['Authorization'] = `Bearer ${apiToken}`;
        }
    }
    
    /**
     * Make HTTP request with error handling.
     * 
     * @param {string} endpoint - API endpoint
     * @param {object} options - Fetch options
     * @returns {Promise<object>} Response data
     */
    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        
        const config = {
            timeout: this.timeout,
            headers: this.headers,
            ...options,
        };
        
        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(`HTTP ${response.status}: ${errorData.detail || response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error('Request timeout');
            }
            throw error;
        }
    }
    
    /**
     * Check API health status.
     * 
     * @returns {Promise<object>} Health status information
     */
    async healthCheck() {
        return await this.makeRequest('/health', {
            method: 'GET',
        });
    }
    
    /**
     * Rate a UDL from content string.
     * 
     * @param {string} content - UDL content to rate
     * @param {string|null} filename - Optional filename
     * @param {boolean} useCtm - Whether to use CTM model
     * @param {boolean} includeTrace - Whether to include computation trace
     * @returns {Promise<object>} Rating response
     */
    async rateUdl(content, filename = null, useCtm = false, includeTrace = false) {
        const payload = {
            content,
            filename,
            use_ctm: useCtm,
            include_trace: includeTrace,
        };
        
        return await this.makeRequest('/rate', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    }
    
    /**
     * Rate a UDL from file.
     * 
     * @param {File} file - File object containing UDL content
     * @param {boolean} useCtm - Whether to use CTM model
     * @param {boolean} includeTrace - Whether to include computation trace
     * @returns {Promise<object>} Rating response
     */
    async rateUdlFile(file, useCtm = false, includeTrace = false) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('use_ctm', useCtm.toString());
        formData.append('include_trace', includeTrace.toString());
        
        // Remove Content-Type header for FormData
        const headers = { ...this.headers };
        delete headers['Content-Type'];
        
        return await this.makeRequest('/rate/file', {
            method: 'POST',
            headers,
            body: formData,
        });
    }
    
    /**
     * Rate multiple UDLs in batch.
     * 
     * @param {Array<object>} udls - List of UDL requests
     * @param {boolean} parallel - Whether to process in parallel
     * @returns {Promise<object>} Batch rating response
     */
    async rateUdlBatch(udls, parallel = true) {
        const payload = {
            udls,
            parallel,
        };
        
        return await this.makeRequest('/rate/batch', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    }
    
    /**
     * Get list of available quality metrics.
     * 
     * @returns {Promise<object>} Information about available metrics
     */
    async getAvailableMetrics() {
        return await this.makeRequest('/metrics', {
            method: 'GET',
        });
    }
    
    /**
     * Rate multiple UDL files from file input.
     * 
     * @param {FileList} files - List of files to rate
     * @param {boolean} useCtm - Whether to use CTM model
     * @param {boolean} includeTrace - Whether to include computation trace
     * @param {boolean} parallel - Whether to process in parallel
     * @returns {Promise<Array<object>>} List of rating results
     */
    async rateFiles(files, useCtm = false, includeTrace = false, parallel = true) {
        const udlRequests = [];
        
        // Read all files
        for (const file of files) {
            try {
                const content = await this.readFileAsText(file);
                udlRequests.push({
                    content,
                    filename: file.name,
                    use_ctm: useCtm,
                    include_trace: includeTrace,
                });
            } catch (error) {
                console.warn(`Could not read ${file.name}:`, error);
            }
        }
        
        if (udlRequests.length === 0) {
            return [];
        }
        
        // Process in batches of 10
        const batchSize = 10;
        const allResults = [];
        
        for (let i = 0; i < udlRequests.length; i += batchSize) {
            const batch = udlRequests.slice(i, i + batchSize);
            try {
                const batchResponse = await this.rateUdlBatch(batch, parallel);
                allResults.push(...batchResponse.results);
            } catch (error) {
                console.warn(`Batch ${Math.floor(i / batchSize) + 1} failed:`, error);
            }
        }
        
        return allResults;
    }
    
    /**
     * Read file as text.
     * 
     * @param {File} file - File to read
     * @returns {Promise<string>} File content as text
     */
    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(e);
            reader.readAsText(file);
        });
    }
}

/**
 * Custom error class for UDL Rating Client.
 */
class UDLRatingError extends Error {
    constructor(message) {
        super(message);
        this.name = 'UDLRatingError';
    }
}

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UDLRatingClient, UDLRatingError };
}

// Example usage (browser)
if (typeof window !== 'undefined') {
    window.UDLRatingClient = UDLRatingClient;
    window.UDLRatingError = UDLRatingError;
    
    // Example usage
    async function exampleUsage() {
        // Initialize client
        const client = new UDLRatingClient(
            'http://localhost:8000',
            null // Set API token if required
        );
        
        try {
            // Check health
            const health = await client.healthCheck();
            console.log('API Status:', health.status);
            console.log('Model Loaded:', health.model_loaded);
            
            // Example UDL content
            const udlContent = `
                grammar SimpleCalculator {
                    expr = term (('+' | '-') term)*
                    term = factor (('*' | '/') factor)*
                    factor = number | '(' expr ')'
                    number = [0-9]+
                }
            `;
            
            // Rate the UDL
            const result = await client.rateUdl(
                udlContent,
                'simple_calculator.udl',
                false, // use_ctm
                true   // include_trace
            );
            
            console.log('Overall Score:', result.overall_score.toFixed(3));
            console.log('Confidence:', result.confidence.toFixed(3));
            console.log('Processing Time:', result.processing_time.toFixed(3) + 's');
            console.log('Model Used:', result.model_used);
            
            console.log('Metric Scores:');
            result.metrics.forEach(metric => {
                console.log(`  ${metric.name}: ${metric.value.toFixed(3)}`);
            });
            
            if (result.trace) {
                console.log(`Computation Trace: ${result.trace.length} steps`);
            }
            
            // Get available metrics
            const metricsInfo = await client.getAvailableMetrics();
            console.log(`Available Metrics: ${metricsInfo.metrics.length}`);
            metricsInfo.metrics.forEach(metric => {
                console.log(`  - ${metric.name}`);
            });
            
        } catch (error) {
            console.error('Error:', error.message);
        }
    }
    
    // Run example if in browser console
    // exampleUsage();
}