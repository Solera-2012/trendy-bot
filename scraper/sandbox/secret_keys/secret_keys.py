

class secret_keys():

	consumer_key='cejLPJb9SjGBLnk4MOH9DiwOq'
	consumer_secret='DwzLoVik2OjVJNhiPWBfg7pXXvwJ5F8IpLW0Amf51Wh64SQBUK'
	access_token_key='4800875312-K5ZPodvOKdJ2q4KC6f9slVemYl7XZQG2g2JoVCs'
	access_token_secret='UxDaAEvLBRzDWvLyv3VmG1nBTYgkxr775WFoaiH6cvQOB'

	def __init__(self):
		pass

	def __repr__(self):
		return "\
		consumer key: %s\n \
		consumer secret: %s\n \
		access token key: %s\n \
		access token secret: %s"% \
		(self.consumer_key, self.consumer_secret, 
		self.access_token_key, self.access_token_secret)




