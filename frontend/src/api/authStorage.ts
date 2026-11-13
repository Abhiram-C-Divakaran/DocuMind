const CSRF_COOKIE_NAME = 'documind_csrf_token';

let inMemoryAccessToken: string | null = null;

export const authStorage = {
  getAccessToken(): string | null {
    return inMemoryAccessToken;
  },
  setAccessToken(accessToken: string | null): void {
    inMemoryAccessToken = accessToken;
  },
  getCsrfToken(): string | null {
    const prefix = `${CSRF_COOKIE_NAME}=`;
    const parts = document.cookie ? document.cookie.split('; ') : [];
    for (const part of parts) {
      if (part.startsWith(prefix)) {
        return decodeURIComponent(part.slice(prefix.length));
      }
    }
    return null;
  },
  clear(): void {
    inMemoryAccessToken = null;
  },
};

