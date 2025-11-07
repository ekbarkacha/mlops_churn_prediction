#!/bin/bash

# Test script for the refactored API
# Usage: bash test_api.sh

API_URL="http://localhost:8000"
API_KEY="your-api-key"

echo "🧪 Testing Customer Churn Prediction API - Refactored Version"
echo "=============================================================="

# Test 1: Root endpoint
echo ""
echo "1️⃣ Testing root endpoint..."
curl -s -X GET "$API_URL/" | jq .

# Test 2: Health check
echo ""
echo "2️⃣ Testing health check..."
curl -s -X GET "$API_URL/health" | jq .

# Test 3: Metrics endpoint
echo ""
echo "3️⃣ Testing metrics endpoint..."
curl -s -X GET "$API_URL/metrics" | head -n 10

# Test 4: OpenAPI docs
echo ""
echo "4️⃣ Testing OpenAPI schema..."
curl -s -X GET "$API_URL/openapi.json" | jq '.info'

echo ""
echo "✅ Basic tests completed!"
echo ""
echo "📚 For interactive testing, visit:"
echo "   - Swagger UI: $API_URL/docs"
echo "   - ReDoc: $API_URL/redoc"
echo ""
echo "🔐 To test authenticated endpoints, you need:"
echo "   1. API Key in header: x-api-key"
echo "   2. JWT token in cookie: access_token"
