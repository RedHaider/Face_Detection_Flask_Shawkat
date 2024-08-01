from datetime import timedelta, datetime, timezone
from threading import Lock
import requests
from ai_api_client_sdk.exception import AIAPIAuthenticatorException, AIAPIAuthenticatorInvalidRequestException, \
    AIAPIAuthenticatorAuthorizationException, AIAPIAuthenticatorServerException, \
    AIAPIAuthenticatorForbiddenException, AIAPIAuthenticatorMethodNotAllowedException, \
    AIAPIAuthenticatorTimeoutException

PARAM_ERROR_MESSAGE = 'Either client_secret needs to provided, or both cert_file_path and key_file_path'


class Authenticator:
    """Authenticator class is implemented to retrieve and cache the authorization token from the xsuaa server

    :param auth_url: URL of the authorization endpoint. Should be the full URL (including /oauth/token)
    :type auth_url: str
    :param client_id: client id to be used for authorization
    :type client_id: str
    :param client_secret: client secret to be used for authorization, either client_secret or
        (cert_file_path and key_file_path) need to be provided, defaults to None
    :type client_secret: str, optional
    :param cert_file_path: path to the certificate file, needs to be provided alongside the key_file_path parameter,
        defaults to None
    :type cert_file_path: str, optional
    :param key_file_path: path to the key file, needs to be provided alongside the cert_file_path parameter,
        defaults to None
    :type key_file_path: str, optional
    """

    def __init__(self, auth_url: str, client_id: str, client_secret: str = None, cert_file_path: str = None,
                 key_file_path: str = None):
        self.url: str = auth_url
        self.client_id: str = client_id
        self.client_secret: str = client_secret
        self.cert_file_path: str = cert_file_path
        self.key_file_path: str = key_file_path
        if not ((client_secret is not None and cert_file_path is None and key_file_path is None)
                or (client_secret is None and cert_file_path is not None and key_file_path is not None)):
            raise AIAPIAuthenticatorException(error_message=PARAM_ERROR_MESSAGE)
        self.token = None # Token Caching
        self.token_expiry_date = None
        self.lock = Lock() # Thread-Safe Lock

    def get_token(self) -> str:
        """Retrieves the token from the xsuaa server or from cache when expiration date not reached.

        :raises: class:`ai_api_client_sdk.exception.AIAPIAuthenticatorException` if an unexpected exception occurs while
            trying to retrieve the token
        :return: The Bearer token
        :rtype: str
        """
        with self.lock: # Thread-Safe
            if self._should_refresh_token():
                data = {'grant_type': 'client_credentials', 'client_id': self.client_id}
                if self.client_secret:
                    data['client_secret'] = self.client_secret
                error_msg = None
                try:
                    if self.client_secret:
                        response = requests.post(url=self.url, data=data)
                    elif self.cert_file_path and self.key_file_path:
                        response = requests.post(url=self.url, data=data,
                                                 cert=(self.cert_file_path, self.key_file_path))
                    else:
                        raise AIAPIAuthenticatorException(error_message=PARAM_ERROR_MESSAGE)
                    status_code = response.status_code
                    error_msg = response.text
                except AIAPIAuthenticatorException as authenticator_exception:
                    raise authenticator_exception
                except Exception as exception:  # pylint:disable=broad-except
                    raise AIAPIAuthenticatorException(status_code=500, error_message=error_msg) from exception

                if status_code == 400:
                    raise AIAPIAuthenticatorInvalidRequestException(error_message=error_msg)
                elif status_code == 401:
                    raise AIAPIAuthenticatorAuthorizationException(error_message=error_msg)
                elif status_code == 403:
                    raise AIAPIAuthenticatorForbiddenException(error_message=error_msg)
                elif status_code == 405:
                    raise AIAPIAuthenticatorMethodNotAllowedException(error_message=error_msg)
                elif status_code == 408:
                    raise AIAPIAuthenticatorTimeoutException(error_message=error_msg)
                elif status_code // 100 != 2:
                    raise AIAPIAuthenticatorServerException(status_code=status_code, error_message=error_msg)

                try:
                    access_token = response.json()['access_token']
                    self.token = f'Bearer {access_token}'
                    self._calc_token_expiry_date(response.json()['expires_in'])
                except Exception as exception:  # pylint:disable=broad-except
                    raise AIAPIAuthenticatorException(status_code=500, error_message=error_msg) from exception
        return self.token

    def _should_refresh_token(self):
        if self.token is None or self.token_expiry_date is None:
            return True

        now = datetime.now(timezone.utc)
        # Check if token has expired incl. buffer
        return self.token_expiry_date - now < timedelta(minutes=60)

    def _calc_token_expiry_date(self, expires_in: str):
        now = datetime.now(timezone.utc)
        # Calculate the token expiry date starting now adding expires in
        self.token_expiry_date = now + timedelta(seconds=int(expires_in))
