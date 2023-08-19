from prefect_email import EmailServerCredentials

credentials = EmailServerCredentials(
    username="####@gmail.com",
    password="",  # must be an app password
)
credentials.save("emailnotification", overwrite=True)
