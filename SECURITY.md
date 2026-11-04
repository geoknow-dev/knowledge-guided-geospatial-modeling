# Security

Do not commit API keys, tokens, passwords, private endpoints or local filesystem paths. Credentials must be supplied through environment variables. The `.env` file is ignored by Git.

If a credential has ever been committed or shared, revoke it at the provider immediately and create a new one. Removing it from the latest file is not sufficient because Git history retains old content.

