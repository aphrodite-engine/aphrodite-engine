# Aphrodite Router

Aphrodite Router is an intelligent request routing system for Aphrodite Engine that directs OpenAI API requests to the most suitable models based on semantic understanding of request intent, complexity, and task type. The router integrates with Envoy Proxy via the ExtProc protocol to provide transparent routing with minimal latency overhead.

## Architecture

The router consists of three main components:
- a Go-based routing server that implements the ExtProc protocol,
- Rust bindings for ML model inference, and
- a configuration system for defining routing rules and model mappings.

The Go router handles request processing, classification, and routing decisions, while the Rust library provides efficient inference for category classification, PII detection, and semantic similarity calculations.

## Building

The router requires both Rust and Go to be installed. Build the Rust library first by running:

```bash
cd candle_binding
cargo build --release
```

This produces the shared library that the Go router links against. Then build the Go router by running:
```bash
cd ../router
go build -o aphrodite-router ./cmd/main.go
```
Alternatively, use the provided `quickstart.sh` script which automates the build process.

The router can also be built using Docker. The `Dockerfile.extproc` provides a multi-stage build that compiles both the Rust library and Go binary into a container image.

## Configuration

Configuration is managed through YAML files in `router/config/`. The main configuration file defines Aphrodite backend endpoints, routing categories, model mappings, and feature flags. Endpoints are specified with their network address and port. Categories define routing rules that map request types to preferred models, with optional system prompts and reasoning mode settings.

The router supports keyword-based classification as a fallback when ML models are not available. Keyword rules can be defined in the configuration to enable basic routing without requiring model files. Model configuration maps logical model names to backend endpoints and defines reasoning families for models that support chain-of-thought reasoning.

### Configuration Structure

The configuration file is organized into several key sections:

- **aphrodite_endpoints**: Defines the backend Aphrodite API instances with their network addresses, ports, and health check settings
- **model_config**: Maps logical model names to physical endpoints and configures reasoning families for models that support chain-of-thought
- **categories**: Defines routing categories with model scoring, system prompts, and optional reasoning mode settings
- **keyword_rules**: Provides keyword-based classification rules for basic routing without ML models
- **classifier**: Configures ML-based classification models (optional, can be disabled)
- **semantic_cache**: Configures semantic caching for improved latency
- **prompt_guard**: Security features for PII detection and jailbreak protection

### Example Configuration

Here's a minimal example configuration for routing between two models:

```yaml
# Define backend endpoints
aphrodite_endpoints:
  - name: "Qwen/Qwen3-4B-Instruct-2507"
    address: "127.0.0.1"
    port: 2242
    weight: 1
  
  - name: "Qwen/Qwen3-14B"
    address: "127.0.0.1"
    port: 2243
    weight: 1

# Map logical model names to endpoints
model_config:
  "Qwen/Qwen3-4B-Instruct-2507-instruct":
    preferred_endpoints: ["Qwen/Qwen3-4B-Instruct-2507"]
    reasoning_family: "qwen3"
  
  "Qwen/Qwen3-14B-a3b-instruct":
    preferred_endpoints: ["Qwen/Qwen3-14B"]
    reasoning_family: "qwen3"

# Define routing categories
categories:
  - name: "general"
    keyword_rules:
      - patterns: [".*"]
        match_type: "regex"
    model_scores:
      "Qwen/Qwen3-4B-Instruct-2507-instruct": 1.0
      "Qwen/Qwen3-14B-a3b-instruct": 0.8

# Set default model for fallback routing
default_model: "Qwen/Qwen3-4B-Instruct-2507-instruct"
```

This configuration enables keyword-based routing where all requests are classified as "general" and routed to the 4B model by default, with the 14B model as a secondary option. The router will automatically select the model based on the configured scores when a request arrives with the auto model name.

### Environment Variables

Environment variables are managed through a `.env` file in the project root. Copy `.env.example` to `.env` and customize the values for your deployment. The dashboard backend automatically loads environment variables from `.env` on startup. These variables control dashboard configuration, router API endpoints, and optional service integrations like Grafana, Prometheus, and Jaeger.

## Running

The router runs as an ExtProc server that integrates with Envoy Proxy. Start the router with:

```bash
./aphrodite-router.sh --config config/config.yaml
```
The router exposes three services: an ExtProc gRPC server on port 50051 for Envoy integration, a Classification API on port 8080 for standalone classification requests, and a metrics endpoint on port 9190 for Prometheus scraping.

For production deployments, configure Envoy to use the router as an external processor. The Envoy configuration file at `router/config/envoy.yaml` demonstrates the integration. Envoy listens on port 8801 by default and forwards requests through the ExtProc filter to the router on port 50051. The router analyzes requests, selects appropriate models, and sets routing headers that Envoy uses to forward requests to the correct Aphrodite backend.

## Routing Logic

When a request arrives with the auto model name (default "MoM"), the router classifies the request content using keyword matching or ML-based classification if available. Classification determines the request category, which maps to preferred models based on configured scores. The router selects the highest-scoring model for that category and routes the request to the corresponding endpoint. The router can also inject category-specific system prompts and enable reasoning mode for models that support it.

For non-auto model requests, the router validates the model name against configuration and routes directly to the appropriate endpoint. PII detection and prompt guard features can block requests that violate security policies.

## Dashboard

A web-based dashboard is available for monitoring and configuration management. The dashboard consists of a React frontend and Go backend that provides a unified interface for viewing router configuration, monitoring metrics, and managing settings. To build and run the dashboard:

Build the frontend by running:
```bash
cd dashboard/frontend
npm install
npm run build
```
Start the backend with:
```bash
cd dashboard/backend
go run main.go
```
The dashboard runs on port 8700 by default and proxies to Grafana, Prometheus, and the router API based on environment variable configuration

The dashboard supports optional integrations with Grafana, Prometheus, Jaeger, Open WebUI, and Hugging Face Chat UI. Configure these services by setting the appropriate environment variables in your `.env` file. If a service is not configured, the corresponding dashboard page will display a message indicating the service is unavailable.

## Development

The project structure separates concerns into distinct packages:
- The `router/pkg/extproc` package contains the ExtProc server implementation and request handling logic.
- The `router/pkg/classification` package provides classification abstractions that support multiple backends including ML models, keyword matching, and MCP services.
- The `router/pkg/config` package handles configuration loading and validation.

Tests can be run with:
```bash
cd router
go test ./...
```
The Rust library includes its own test suite that can be run with:
```bash
cargo test
```

## Dependencies

The router requires Go 1.25.3 or higher and Rust 1.90 or higher. Runtime dependencies include Envoy Proxy for production deployments, though the router can run standalone for testing. The Rust library uses Candle for ML inference and requires the shared library to be available at runtime, either through the system library path or via the wrapper script that sets `LD_LIBRARY_PATH`.

