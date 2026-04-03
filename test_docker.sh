docker rm -f br-smoke || true
docker run -d -p 8000:8000 --name br-smoke budget-router-test
sleep 10
curl -s -X POST http://localhost:8000/reset | python3 -m json.tool
docker stop br-smoke && docker rm br-smoke
