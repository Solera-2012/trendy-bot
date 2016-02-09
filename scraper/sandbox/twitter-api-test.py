from secret_keys import secret_keys


key = secret_keys.secret_keys()
print(key.consumer_key)
print(key.consumer_secret)
print(key.access_token_key)
print(key.access_token_secret)
