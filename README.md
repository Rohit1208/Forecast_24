# MarketForecast

A role-based Streamlit web application for forecasting next-bar NSE OHLC values using fresh Yahoo Finance history and the official NSE equity directory.

## Run locally

\`\`\`powershell
python -m pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Roles and accounts

- The first account created becomes the administrator. In production, set \`ADMIN_EMAIL\`, \`ADMIN_PASSWORD\`, and optionally \`ADMIN_NAME\` before the first start to create the owner account explicitly.
- All later registrations are held as **pending**. An administrator can approve each account as a **user** or **administrator**.
- Password-reset requests are presented to the administrator, who can issue a new temporary password through the protected admin view.
- Passwords are stored as salted PBKDF2 hashes in SQLite—never as plain text.

The database is stored at \`data/marketforecast.db\` by default. For deployment, set \`APP_DB_PATH\` to a persistent protected disk location. A single-instance EC2 deployment can use the attached EBS volume; do not rely on a container's temporary filesystem for production user accounts.

## Forecasting behavior

Users can select up to ten NSE stocks and one interval. The app evaluates XGBoost, Gradient Boosting, Histogram Gradient Boosting, and Ridge Regression using the latest five walk-forward validations, automatically uses the lowest recent close-deviation model, and shows only the forecast results and validation data. Algorithm controls and charts are restricted to the administrator workspace. Recent identical forecasts are cached for five minutes to speed up repeat requests.

Yahoo Finance may delay or revise market data. Forecasts are estimates for informational use only and are not investment advice.

## AWS deployment

The included \`Dockerfile\` runs the service on port \`8501\`. On EC2, allow port 8501 only while testing; for a public production site, place it behind HTTPS via a reverse proxy/load balancer and use a persistent database. Do not commit passwords, AWS keys, or email credentials to this repository.

