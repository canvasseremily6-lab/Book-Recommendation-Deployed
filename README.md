# Book Recommendation FastAPI Application, Streamlit Dashboard, and Streamlit Frontend

An end-to-end ML system that recommends books based on a user's favorite title. A Naive Bayes model predicts a genre from the book's title + description, then the app recommends two books from that genre. This porject includes a model registry, a FastAPI backend, a Streamlit frontend, a cloud database (DynamoDB) from aws for logging/caching, a live monitoring Streamlit dashboard, automated testing, and containerized deplotment to AWS EC2. 

## Prerequisites 

**Prerequisites:** Python 3.12, Docker Desktop, an AWS account (or AWS Academy Learner Lab), a Weights & Biases account.

- Make sure that you have downloaded the requirements.txt

## How to Run 

- **Clone the Repository** to your local machine: 
'''bash
git clone https://github.com/canvasseremily6-lab/Book-Recommendation-Deployed
cd Book-Recommendation-Deployed

python3 -m venv venv
source venv/bin/activate
'''

- **Install** the required libraries: 
'''bash
pip install -r requirements.txt 
'''

- **Configure credentials:**
 
1. **Weights & Biases:**
```bash
   wandb login
```
2. **AWS** (for DynamoDB access):
```bash
   aws configure
```
   or manually set `~/.aws/credentials` and `~/.aws/config` (region: `us-east-1`).

- **Run**  
## Running Locally
 
**1. Train the model** (downloads the dataset, trains, logs to W&B, promotes to the registry):
 
```bash
python train_model.py
```
 
**2. Create the DynamoDB tables** (one-time):
 
```bash
python create_tables.py
```
 
**3. Start the API backend:**
 
```bash
python -m uvicorn main:app --reload
```
API available at `http://127.0.0.1:8000` — interactive docs at `http://127.0.0.1:8000/docs`.
 
**4. Start the frontend** (in a separate terminal):
 
```bash
python -m streamlit run app.py
```
Available at `http://localhost:8501`.
 
**5. Start the dashboard** (in a separate terminal):
 
```bash
python -m streamlit run dashboard.py --server.port 8502
```
Available at `http://localhost:8502`.

## Docker
 
Each service has its own Dockerfile, built and run independently.
 
```bash
# API
docker build -f Dockerfile.api -t book-api .
docker run -d -p 8000:8000 --env-file .env --restart unless-stopped book-api
 
# Frontend
docker build -f Dockerfile.app -t book-frontend .
docker run -d -p 8501:8501 -e API_URL=http://54.158.50.133:8000 --restart unless-stopped book-frontend
 
# Dashboard
docker build -f Dockerfile.dashboard -t book-dashboard .
docker run -d -p 8502:8502 --env-file .env --restart unless-stopped book-dashboard
```
 
`.env` (required for API and Dashboard containers — **never commit this file**):
 
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
WANDB_API_KEY=...
```
 
 
## AWS Deployment
 
Deployed to **three separate EC2 instances** (Amazon Linux 2023, t2.micro), each running one Dockerized service.
 
**Per-instance setup (repeat for API, frontend, and dashboard servers):**
 
```bash
# 1. SSH in
ssh -i book-app-key.pem ec2-user@13.218.12.75
 
# 2. Install Docker + git
sudo yum update -y
sudo yum install docker git -y
sudo service docker start
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user
exit
# reconnect for group permissions to apply
ssh -i book-app-key.pem ec2-user@13.218.12.75D
 
# 3. Clone the repo
git clone https://github.com/canvasseremily6-lab
cd Book-Recommendation-Deployed
 
# 4. Create .env (API and dashboard instances only)
nano .env
 
# 5. Build and run the correct container for this instance
docker build -f Dockerfile.<api|app|dashboard> -t book-<name> .
docker run -d -p <PORT>:<PORT> --env-file .env --restart unless-stopped book-<name>
```
 
**Security groups:** each instance's inbound rules allow SSH (22) from your IP, plus the service's port (8000 / 8501 / 8502) from `0.0.0.0/0`.
 
**Redeploying after a code change:**
 
```bash
git pull origin main
docker stop $(docker ps -q --filter ancestor=book-<name>)
docker rm $(docker ps -aq --filter ancestor=book-<name>)
docker build -f Dockerfile.<name> -t book-<name> .
docker run -d -p <PORT>:<PORT> --env-file .env --restart unless-stopped book-<name>

- ** Note: this project was created using AWS Academy Learner Lab** credentials are temporary and expire every few hours. If you see 'Unable to locate credentials' error, then you will need to refresh the .env with the fresh values and restart the container. 

## Testing
 
```bash
python -m pytest -v
```
 
- **Unit tests** — `clean_category` and `map_genres` (preprocessing logic), covering list-string parsing, case-insensitivity, and edge cases (`None`, `NaN`, unmatched categories).
- **Integration tests** — FastAPI endpoints (`/`, `/health`, `/predict`) using `TestClient`, with the model, book data, and DynamoDB mocked out so tests run fast and don't require live credentials.
---
 
## CI/CD
 
`.github/workflows/ci.yml` runs automatically on every pull request to `main`:
 
1. Installs dependencies from `requirements.txt`
2. Lints with `ruff`
3. Runs the full `pytest` suite
 
 
## Example: API Requests
 
**Health check:**
```bash
curl http://54.158.50.133:8000/health
```
```json
{"status": "ok", "model_loaded": true, "books_loaded": true}
```
 
**Get a recommendation:**
```bash
curl -X POST http://54.158.50.133:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Harry Potter and the Sorcerer'\''s Stone"}'
```
```json
{
  "genre": "Fiction",
  "rec_1": "Some Book Title",
  "rec_2": "Another Book Title",
  "cached": false,
  "request_id": "16fccd90-6da0-4760-ada0-5f99b1677b68"
}
```
Calling `/predict` again with the **same title** returns the cached result (`"cached": true`) without re-running.
 
**Submit feedback on a recommendation:**
```bash
curl -X POST http://54.158.50.133:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"request_id": "16fccd90-6da0-4760-ada0-5f99b1677b68", "is_good_recommendation": true}'
```
```json
{"status": "feedback recorded"}
```
 
 
## Monitoring Dashboard
 
The dashboard (`http://54.197.202.49:8502`) reads directly from the `prediction_logs` DynamoDB tables and visualizes:
 
- **Prediction latency over time** — line chart plus average / min / max
- **Distribution of predicted genres (target drift)** — overall bar chart, and a time-series breakdown to spot drift across days
- **Live accuracy from user feedback** — 'yes'/'no' collected in the frontend, aggregated into a rolling accuracy percentage