
import time
import jwt  # Ensure you're using PyJWT

def is_token_expired(token, *args, tolerance=300):
    """
    Decode the JWT token and check if it's near expiration.
    Returns True if the token is expired or will expire within the given tolerance.
    """
    try:
        if isinstance(token, bytes):
            token = token.decode('utf-8')

        expected_audience = "letsdoit-3924b"  # Adjust to match your Firebase project ID

        decoded_token = jwt.decode(
            token,
            audience=expected_audience,
            options={"verify_signature": False, "verify_iat": False}  # ignoring signature for local checks
        )

        exp = decoded_token.get("exp", 0)  # in seconds since epoch
        current_time = int(time.time())

        # Check if token is near expiry (within tolerance)
        return current_time >= (exp - tolerance)
    except jwt.InvalidAudienceError:
        print("Error decoding token: Invalid audience")
        return True
    except Exception as e:
        print(f"Error decoding token: {e}")
        return True
