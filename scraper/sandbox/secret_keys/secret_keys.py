

class secret_keys():

	consumer_key=''
	consumer_secret=''
	access_token_key=''
	access_token_secret=''

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




