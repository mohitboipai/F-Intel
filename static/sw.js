const CACHE_NAME = 'fintel-v2';
const CACHED_URLS = [
  '/',
  '/static/manifest.json',
  '/static/icon.png'
];

self.addEventListener('install', event => {
  // Force immediate activation
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(CACHED_URLS);
    })
  );
});

self.addEventListener('activate', event => {
  // Claim clients immediately and purge old caches
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  // Never intercept WebSocket
  if (event.request.url.startsWith('ws://') || event.request.url.startsWith('wss://')) {
    return;
  }
  
  const url = new URL(event.request.url);

  // Network-First for root dashboard page so updates appear instantly
  if (url.pathname === '/') {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // Check if it's one of the other cached URLs (manifest, icons)
  if (CACHED_URLS.includes(url.pathname)) {
    event.respondWith(
      caches.match(event.request).then(response => {
        return response || fetch(event.request);
      })
    );
  } else {
    // Network only for fragments, APIs, and stream
    event.respondWith(fetch(event.request));
  }
});
