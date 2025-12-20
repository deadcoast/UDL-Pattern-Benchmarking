# UDL Rating Framework Deployment

This directory contains all the necessary files and configurations for deploying the UDL Rating Framework as a production-ready service.

## Overview

The deployment supports multiple deployment strategies:

1. **Docker Compose** - For local development and testing
2. **Kubernetes** - For production deployment with auto-scaling
3. **Standalone Docker** - For simple containerized deployment

## Features

- **REST API** with FastAPI for high-performance web service
- **Rate limiting** and authentication for security
- **Auto-scaling** with Kubernetes HPA
- **Monitoring** with Prometheus and Grafana
- **Load balancing** with Nginx reverse proxy
- **Multi-stage Docker builds** for optimized images
- **Health checks** and graceful shutdown
- **TLS/SSL support** for secure communication

## Quick Start

### Docker Compose (Recommended for Development)

1. **Build and start services:**
   ```bash
   cd deployment/docker
   docker-compose up -d
   ```

2. **Access the API:**
   - API: http://localhost:8000
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

3. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   ```

### Kubernetes (Production)

1. **Prerequisites:**
   - Kubernetes cluster (1.20+)
   - kubectl configured
   - Docker registry access

2. **Build and push image:**
   ```bash
   ./deployment/scripts/build.sh
   PUSH_TO_REGISTRY=true ./deployment/scripts/build.sh
   ```

3. **Deploy to Kubernetes:**
   ```bash
   ./deployment/scripts/deploy.sh kubernetes
   ```

4. **Check deployment:**
   ```bash
   kubectl get pods -n udl-rating
   kubectl get services -n udl-rating
   ```

## Directory Structure

```
deployment/
├── api/                    # FastAPI application
│   ├── main.py            # Main API application
│   └── requirements.txt   # API dependencies
├── docker/                # Docker configuration
│   ├── Dockerfile         # Multi-stage Docker build
│   ├── docker-compose.yml # Local development setup
│   ├── nginx.conf         # Nginx configuration
│   └── prometheus.yml     # Prometheus configuration
├── kubernetes/            # Kubernetes manifests
│   ├── namespace.yaml     # Namespace definition
│   ├── configmap.yaml     # Configuration maps
│   ├── secret.yaml        # Secrets (template)
│   ├── deployment.yaml    # Application deployments
│   ├── service.yaml       # Service definitions
│   ├── hpa.yaml          # Horizontal Pod Autoscaler
│   ├── ingress.yaml      # Ingress configuration
│   ├── pvc.yaml          # Persistent Volume Claims
│   └── monitoring.yaml   # Monitoring stack
├── scripts/              # Deployment scripts
│   ├── build.sh         # Build Docker images
│   ├── deploy.sh        # Deploy to environment
│   └── cleanup.sh       # Cleanup deployment
├── client/              # API clients
│   ├── python_client.py # Python client library
│   └── javascript_client.js # JavaScript client
└── README.md           # This file
```

## API Endpoints

### Core Endpoints

- `GET /health` - Health check
- `POST /rate` - Rate UDL from content
- `POST /rate/file` - Rate UDL from file upload
- `POST /rate/batch` - Rate multiple UDLs
- `GET /metrics` - Get available metrics

### Example Usage

**Rate a UDL:**
```bash
curl -X POST "http://localhost:8000/rate" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "grammar Test { rule = \"hello\" }",
    "filename": "test.udl",
    "include_trace": true
  }'
```

**Upload and rate a file:**
```bash
curl -X POST "http://localhost:8000/rate/file" \
  -F "file=@example.udl" \
  -F "use_ctm=false" \
  -F "include_trace=true"
```

## Configuration

### Environment Variables

#### API Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `API_TOKEN` | Authentication token | None (no auth) |
| `CTM_MODEL_PATH` | Path to CTM model file | `/app/models/ctm_model.pt` |
| `ENVIRONMENT` | Environment (development/production) | `production` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `PORT` | API port | `8000` |

#### Coverage Monitoring (CI/CD)

These variables are used by the coverage monitoring script (`scripts/coverage_monitor.py`) for sending email alerts:

| Variable | Description | Default |
|----------|-------------|---------|
| `SMTP_SERVER` | SMTP server hostname for email alerts | `localhost` |
| `SMTP_PORT` | SMTP server port | `587` |
| `SMTP_USER` | SMTP authentication username | None |
| `SMTP_PASSWORD` | SMTP authentication password | None |
| `COVERAGE_ALERT_RECIPIENTS` | Comma-separated list of email recipients for coverage alerts | None |

#### Docker Compose Specific

| Variable | Description | Default |
|----------|-------------|---------|
| `GRAFANA_PASSWORD` | Grafana admin password | `admin` |

### Docker Compose Configuration

Edit `deployment/docker/docker-compose.yml` to customize:
- Port mappings
- Volume mounts
- Environment variables
- Resource limits

### Kubernetes Configuration

1. **Update secrets** in `deployment/kubernetes/secret.yaml`:
   ```bash
   echo -n "your-api-token" | base64
   ```

2. **Configure ingress** in `deployment/kubernetes/ingress.yaml`:
   - Update hostname
   - Configure TLS certificates

3. **Adjust resources** in `deployment/kubernetes/deployment.yaml`:
   - CPU/memory requests and limits
   - Replica counts

## Security

### Authentication

Set the `API_TOKEN` environment variable to enable bearer token authentication:

```bash
export API_TOKEN="your-secure-token"
```

All API requests must include the token:
```bash
curl -H "Authorization: Bearer your-secure-token" http://localhost:8000/rate
```

### Rate Limiting

Built-in rate limiting per IP address:
- `/rate`: 10 requests/minute
- `/rate/file`: 5 requests/minute  
- `/rate/batch`: 2 requests/minute

### TLS/SSL

For production deployment:

1. **Docker Compose**: Place certificates in `deployment/docker/ssl/`
2. **Kubernetes**: Update TLS secret in `deployment/kubernetes/secret.yaml`

## Monitoring

### Prometheus Metrics

The API exposes metrics at `/metrics` endpoint:
- Request counts and latencies
- Error rates
- Model inference times
- Resource usage

### Grafana Dashboards

Pre-configured dashboards for:
- API performance
- Model metrics
- System resources
- Error tracking

Access Grafana at http://localhost:3000 (admin/admin)

## Scaling

### Horizontal Pod Autoscaler (HPA)

Kubernetes deployment includes HPA configuration:
- **Min replicas**: 3
- **Max replicas**: 10
- **CPU target**: 70%
- **Memory target**: 80%

### Manual Scaling

```bash
kubectl scale deployment udl-rating-api --replicas=5 -n udl-rating
```

## Troubleshooting

### Common Issues

1. **API not responding:**
   ```bash
   # Check container logs
   docker-compose logs udl-rating-api
   
   # Check Kubernetes pods
   kubectl logs -f deployment/udl-rating-api -n udl-rating
   ```

2. **Model not loading:**
   - Verify `CTM_MODEL_PATH` points to valid model file
   - Check model file permissions
   - Review startup logs for errors

3. **Rate limiting issues:**
   - Check Nginx logs for rate limit hits
   - Adjust rate limits in configuration
   - Consider IP whitelisting

4. **Memory issues:**
   - Increase container memory limits
   - Monitor memory usage with Grafana
   - Consider model optimization

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/health | jq .
```

### Log Analysis

```bash
# Docker Compose logs
docker-compose logs -f udl-rating-api

# Kubernetes logs
kubectl logs -f deployment/udl-rating-api -n udl-rating

# Follow logs from all pods
kubectl logs -f -l app=udl-rating-api -n udl-rating
```

## Performance Tuning

### API Performance

1. **Worker processes**: Adjust `--workers` in Dockerfile CMD
2. **Connection pooling**: Configure database connections
3. **Caching**: Implement Redis caching for frequent requests
4. **Async processing**: Use background tasks for heavy computations

### Model Performance

1. **Batch processing**: Use batch endpoints for multiple UDLs
2. **Model optimization**: Quantize or distill models
3. **GPU acceleration**: Add CUDA support for inference
4. **Model caching**: Cache model outputs for identical inputs

## Development

### Local Development

1. **Start development environment:**
   ```bash
   docker-compose -f deployment/docker/docker-compose.yml up -d
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -r deployment/api/requirements.txt
   ```

3. **Run API locally:**
   ```bash
   cd deployment/api
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Testing

```bash
# Run API tests
uv run pytest tests/test_api.py

# Load testing
pip install locust
locust -f tests/load_test.py --host http://localhost:8000
```

## Client Libraries

### Python Client

```python
from deployment.client.python_client import UDLRatingClient

client = UDLRatingClient("http://localhost:8000")
result = client.rate_udl("grammar Test { rule = 'hello' }")
print(f"Score: {result['overall_score']}")
```

### JavaScript Client

```javascript
import { UDLRatingClient } from './deployment/client/javascript_client.js';

const client = new UDLRatingClient('http://localhost:8000');
const result = await client.rateUdl("grammar Test { rule = 'hello' }");
console.log(`Score: ${result.overall_score}`);
```

## Support

For deployment issues:

1. Check the troubleshooting section above
2. Review logs for error messages
3. Verify configuration settings
4. Test with minimal examples
5. Check resource availability (CPU, memory, disk)

## License

This deployment configuration is part of the UDL Rating Framework project.