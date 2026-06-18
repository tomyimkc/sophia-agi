param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,
    [string]$Region = "us-central1",
    [string]$ServiceName = "sophia-rag-api",
    [switch]$UseVertex
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "Building RAG index..."
python tools/build_rag_index.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Image = "gcr.io/$ProjectId/$ServiceName"
Write-Host "Building image $Image ..."
docker build -f services/rag_api/Dockerfile -t $Image .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Pushing image..."
docker push $Image
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$EnvVars = "SOPHIA_RAG_BACKEND=gemini,SOPHIA_RAG_INDEX_DIR=/app/rag/index,GEMINI_MODEL=gemini-2.0-flash"
if ($UseVertex) {
    $EnvVars = "SOPHIA_RAG_BACKEND=vertex,GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=$ProjectId,GOOGLE_CLOUD_LOCATION=$Region,SOPHIA_RAG_INDEX_DIR=/app/rag/index,GEMINI_MODEL=gemini-2.0-flash"
}

Write-Host "Deploying to Cloud Run..."
gcloud run deploy $ServiceName `
    --image $Image `
    --project $ProjectId `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --set-env-vars $EnvVars `
    --memory 1Gi `
    --cpu 1 `
    --min-instances 0 `
    --max-instances 3

if ($UseVertex) {
    Write-Host ""
    Write-Host "Vertex mode: ensure the Cloud Run service account has roles/aiplatform.user"
    Write-Host "For API-key mode instead, create secret GOOGLE_API_KEY and run:"
    Write-Host "  gcloud run services update $ServiceName --update-secrets=GOOGLE_API_KEY=GOOGLE_API_KEY:latest"
}