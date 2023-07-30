from prefect_email import EmailServerCredentials


credentials = EmailServerCredentials(
    username="rehameltagoury@gmail.com",
    password="yiokoznkbbbsfdoe",  # must be an app password
  
)
credentials.save("emailnotification",overwrite=True)