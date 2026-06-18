All the databases are external to the application, and should not be managed by the application. (if in docker-compose, remove all references to them)

Store settings in a json file (settings.json) in the application directory.
Store authentication settings in a local .env file in the application directory.
Default to no authentication and no encryption (TLS) for each server, unless otherwise specified, and provide config key/values for authentication and encryption for each server, so it can be added in the future.

The application uses the following databases. I will provide details about each database below.
- MongoDB database for all data storage.
    - ServerIP:27019
- Redis for caching and session storage.
    - ServerIP:46379
- Clickhouse for all time-series, analytics and logging. - Only install this if the application needs it.
    - ServerIP:8123
- Vector storage, it uses Qdrant - Only install this if the application needs it.
    - ServerIP:6333
- LLM / AI - Models, it uses Ollama - Use LangChain for model management, as they will change in the future. 
    - ServerIP:11434
    - Default model: qwen3.5:4b-q8_0, fallback to gemma4:e2b
    - Embedding model: qwen3-embedding:4b, fallback to nomic-embed-text-v2-moe:latest
    - Include only important model settings: Reasoning, Temperature, Max Tokens, etc.

Create a new database for each application, and prefix the database name with the application name. ie: myapp_mongodb, myapp_redis, etc.
Do not create any databases on these servers that could conflict with existing databases. ie: Users, admin, logs, etc.
Create any indexes, collections, tables, etc. that are needed for the application.
Optimize for performance and scalability. (cluster, replication, etc.)
Do not create generic databases, only create databases that are needed for the application.
