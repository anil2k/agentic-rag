docker run -d --name my_postgres -e POSTGRES_USER=ai -e POSTGRES_PASSWORD=ai -e POSTGRES_DB=ai -p 5532:5432 postgres

# pgadmin
docker run -d --name my_pgadmin -p 5050:80 -e PGADMIN_DEFAULT_EMAIL=admin@example.com -e PGADMIN_DEFAULT_PASSWORD=admin dpage/pgadmin4
Open http://localhost:5050
Login with:
Email: admin@example.com
Password: admin
Add a new server:
Name: my_postgres
Host: host.docker.internal (for Windows) or localhost
Port: 5532
Username: ai
Password: ai
Now you can browse tables, execute SQL queries, and visualize data.