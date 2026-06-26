## Generic PWA Building Prompt 
#-- Updated: 6/25/2026 | Version: 0.0.10

You are a senior full-stack engineer. Upgrade this existing web application into a production-ready Progressive Web App.

Build the PWA in a generic, reusable way so the same implementation pattern can be used across future applications.

Core requirements:

* Add a valid 'manifest.json'
* Add a production-ready service worker
* Add installability support for desktop, Android, iOS/iPadOS, and tablets
* Add offline mode with graceful fallback pages
* Add app icons, favicons, maskable icons, Apple touch icons, and splash screen support
* Add dark mode as the default theme
* Add proper viewport, theme color, background color, and mobile meta tags
* Support mobile, tablet, desktop, and responsive layouts
* Support all major modern browsers where possible

PWA behavior:

* Cache static assets safely
* Cache API responses where appropriate
* Use versioned cache names
* Clean up old caches during service worker activation
* Support offline-first behavior for app shell assets
* Support network-first or stale-while-revalidate strategies for dynamic data
* Add an offline fallback route/page
* Add install prompt handling using 'beforeinstallprompt'
* Detect standalone/PWA mode
* Provide clear UI for “Install App” when supported

Notifications and background features:

* Add Web Push notification support where browser-compatible
* Add permission request flow for notifications
* Add background sync support where available
* Provide safe fallbacks for browsers that do not support background sync or push
* Do not break the app when unsupported PWA APIs are missing
* Provide realtime notification to the user interface via websockets and VAPID keys for web push authorization

Realtime communication:

* Support WebSockets for realtime updates
* Support Server-Sent Events as a fallback
* Support long polling as a final fallback
* Add reconnect logic with exponential backoff
* Detect online/offline status and reconnect when the network returns

File and content input support:

* Support file uploads for images, documents, text files, PDFs, and other allowed files
* Support URL/link submission
* Validate file type, file size, and upload errors
* Show upload progress where possible
* Queue uploads while offline if background sync is supported
* Retry failed uploads safely

Build and deployment:

* Create or update build scripts for the PWA
* Create or update deploy scripts
* Ensure the app can be built for production
* Include cache-busting/versioning for deployed assets
* Ensure HTTPS is required for production PWA features
* Add environment configuration for development, staging, and production
* Document setup, build, deploy, and testing steps

Security and quality:

* Follow secure defaults
* Do not cache sensitive authenticated data unless explicitly safe
* Avoid exposing secrets in frontend code
* Add CSP/security header recommendations
* Validate push subscription and upload endpoints server-side
* Make the implementation accessible and keyboard-friendly
* Add error handling and user-friendly fallback messages
* Disable all spell checking and auto completes on forms which contain password input boxes. 

Testing:

* Test installability with browser dev tools/Lighthouse
* Test offline mode
* Test refresh/update behavior after deployment
* Test icons and splash screens
* Test Android, iOS/iPadOS, desktop Chrome/Edge/Safari/Firefox compatibility
* Test push notification fallback behavior
* Test realtime communication fallback behavior
* Test upload behavior online and offline

Deliverables:

* Updated application code
* 'manifest.json'
* Service worker file
* PWA registration logic
* Install prompt UI
* Offline fallback page
* Icon/favicons/splash-screen integration
* Build script
* Deploy script
* Documentation explaining how to configure, build, test, and deploy the PWA

Important:

* Preserve the existing application behavior.
* Keep the implementation modular and reusable.
* Do not hard-code app-specific names unless clearly marked as configurable.
* Use clean, readable code with comments where helpful.
* Explain any browser limitations, especially for iOS push notifications, background sync, and install prompt behavior.
