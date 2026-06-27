You are updating an existing web application with a modern, production-quality web interface.

Goal:
Create a clean, beautiful, responsive, reusable web UI that works well across desktop, tablet, and mobile devices. The interface should be generic enough to support many types of future FastAPI applications, including dashboards, admin panels, AI tools, data management systems, security tools, monitoring tools, and internal business applications.

Design Requirements:

* Use a modern dark theme by default.
* Use a polished glassmorphism style with translucent panels, subtle borders, soft shadows, blurred backgrounds, and clean spacing.
* Use rounded, soft, clean “bubbly” UI elements that feel modern, friendly, and premium.
* Use a modern SaaS-style dashboard layout with smooth spacing, polished cards, beautiful buttons, and clear visual hierarchy.
* Add a subtle modern background using gradients, abstract patterns, cybersecurity-style visuals, datacenter/network/globe-inspired elements, or animated background accents.
* Keep the background subtle and professional. It should enhance the interface without making text harder to read.
* Make the UI fully responsive for mobile, tablet, and desktop.
* Use accessible colors, readable typography, clear contrast, and consistent spacing.
* Use a modern CSS framework such as Tailwind CSS.
* Create reusable UI components so the interface can be extended easily.

Typography Requirements:

* Use clean, modern, publicly available fonts.
* Prefer open-source/publicly available fonts such as:

  * Inter
  * Manrope
  * Plus Jakarta Sans
  * Outfit
  * Public Sans
  * Space Grotesk
  * IBM Plex Sans
  * Geist
* Use a professional font pairing, for example:

  * Inter or Manrope for body text
  * Plus Jakarta Sans, Outfit, or Space Grotesk for headings
* Use large, bold, clean headings with subtle letter spacing.
* Use readable body text with comfortable line height.
* Use consistent font sizes for headings, labels, buttons, tables, cards, and forms.
* Do not use overly decorative or hard-to-read fonts.
* Load fonts from a public provider such as Google Fonts or use locally bundled open-source font files if the project already has that pattern.
* Provide a safe fallback font stack such as:
  `font-family: "Inter", "Manrope", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;`

Button and Interaction Design Requirements:

* Create beautiful, clean, rounded “bubbly” buttons.
* Buttons should have large rounded corners, soft shadows, subtle gradients, smooth hover states, and clear active/focus states.
* Use pill-shaped buttons where appropriate.
* Add button variants for:

  * Primary action
  * Secondary action
  * Ghost button
  * Danger/delete action
  * Success/confirm action
  * Warning action
  * Icon button
  * Small table-row action button
* Buttons should feel tactile and premium, but not cartoonish.
* Add subtle hover animations such as lift, glow, scale, or gradient movement.
* Add clear focus rings for keyboard accessibility.
* Use icons where helpful, but keep labels clear.
* Destructive buttons should be visually distinct and require confirmation before action.
* Disabled buttons should be clearly muted and non-interactive.
* Keep button styling consistent across the entire app.

Suggested Button Style:

* Rounded corners: `rounded-full` or `rounded-2xl`
* Padding: comfortable spacing such as `px-5 py-2.5`
* Shadows: soft shadows such as `shadow-lg shadow-black/20`
* Borders: subtle translucent borders such as `border border-white/10`
* Backgrounds: gradients or translucent glass backgrounds
* Hover state: slight lift using `hover:-translate-y-0.5`
* Transition: smooth transitions using `transition-all duration-200`
* Focus state: accessible visible ring such as `focus:ring-2 focus:ring-cyan-400/60`

Core Pages / Layout:

* Main dashboard page with cards, charts, quick stats, recent activity, and system status.
* Navigation sidebar for desktop.
* Collapsible/mobile navigation menu for smaller screens.
* Top header bar with page title, global search, theme toggle, user/settings area, and status indicators.
* Settings page for application configuration.
* Database management pages for adding, editing, deleting, searching, filtering, sorting, and viewing records.
* AI model configuration page.
* Logs page for viewing, searching, filtering, and deleting logs.
* Scheduled jobs / automation page for configuring cleanup tasks, background jobs, cron-like schedules, and service maintenance tasks.
* Error pages and empty states that look professional.

UI Component Requirements:
Create reusable components for:

* Layout shell
* Sidebar navigation
* Mobile navigation drawer
* Header/topbar
* Cards
* Stat cards
* Buttons
* Icon buttons
* Forms
* Inputs
* Select menus
* Toggles
* Checkboxes
* Radio buttons
* Search bars
* Tables
* Modals
* Drawers
* Badges
* Alerts
* Toast notifications
* Tabs
* Accordions
* Pagination
* Loading spinners
* Skeleton loaders
* Empty states
* Error states
* Chart containers
* Settings panels

Database Management Requirements:

* Provide a reusable CRUD interface for database records.
* Support add, edit, delete, view, and bulk delete where appropriate.
* Use modals, drawers, or dedicated forms for create/edit actions.
* Add confirmation prompts for destructive actions.
* Add search functionality.
* Add filtering on every table column.
* Add sorting on every table column.
* Add pagination.
* Use DataTables.net or an equivalent modern table solution for table design and functionality.
* Make tables responsive on mobile devices.
* Style tables to match the dark glassmorphism theme.
* Use rounded table containers, soft row hover effects, clear status badges, and small rounded action buttons.
* Show loading states, empty states, and error states.

AI Model Configuration Requirements:

* Add a settings section for configuring AI model providers.
* Support local and remote model providers such as Ollama, vLLM, llama.cpp, OpenAI-compatible APIs, Anthropic, NVIDIA, AMD, MLX, and other future providers.
* Allow the user to define:

  * Provider name
  * Provider type
  * Base URL / host
  * API key or authentication method
  * Default model
  * Primary model
  * Backup/failover model
  * Timeout settings
  * Temperature
  * Max tokens
  * Reasoning settings if supported
  * Enable/disable provider
* Allow a primary AI model and backup AI model for application functions.
* Add failover logic so if the primary host or model fails, the app can automatically use the backup.
* Use LangChain or a clean abstraction layer to manage AI model interactions.
* Keep the AI provider system modular so new providers can be added later.
* Include a “Test Connection” button that checks the provider and lists available models if supported.

Data Visualization Requirements:

* Use Apache ECharts for charts and dashboard visualizations.
* Create reusable chart components.
* Include examples for:

  * Line charts
  * Bar charts
  * Pie/donut charts
  * Area charts
  * Time-series charts
  * Status/health widgets
* Charts should automatically resize and work on mobile.
* Charts should match the dark glassmorphism theme.
* Use rounded glass-style chart containers with clear titles, legends, and tooltips.

Logs Requirements:

* Add a logs page that can view application logs and system/service logs where available.
* Allow log search, filtering, sorting, and pagination.
* Add filters for:

  * Date/time range
  * Log level
  * Source/service
  * Keyword
  * User/action if available
* Allow deleting or clearing logs where appropriate.
* Add confirmation prompts before log deletion.
* Make logs easy to read with color-coded severity levels.
* Use compact, rounded severity badges for DEBUG, INFO, WARNING, ERROR, and CRITICAL.

Scheduled Jobs / Automation Requirements:

* Add a page for managing scheduled application tasks.
* Support cron-style schedules or simple interval-based schedules.
* Allow users to enable, disable, edit, and delete scheduled jobs.
* Include job history, last run time, next run time, status, duration, and result.
* Support jobs such as cleanup tasks, database maintenance, cache cleanup, model refresh, log rotation, report generation, and other future background services.
* Design this in a generic way so new scheduled tasks can be added later.

Backend Integration Requirements:

* Use FastAPI routes cleanly.
* Create or update API endpoints as needed for:

  * CRUD operations
  * Dashboard stats
  * Table data
  * Search/filter/sort/pagination
  * AI provider configuration
  * AI provider test connection
  * Logs
  * Scheduled jobs
  * Settings
* Use Pydantic models for request and response validation.
* Keep backend code modular and organized.
* Do not hardcode application-specific names unless they already exist in the project.
* Preserve existing functionality.
* Avoid breaking existing API routes.
* Add clear error handling and helpful API responses.

Frontend Behavior Requirements:

* Use loading spinners or skeleton loaders where data is loading.
* Show toast notifications for success, warning, and error messages.
* Show clear validation messages on forms.
* Add confirmation dialogs before destructive actions.
* Add reusable components for cards, buttons, forms, tables, charts, modals, drawers, badges, alerts, and settings panels.
* Make the interface feel fast and smooth.
* Use subtle animations and transitions, but avoid anything excessive.
* Support dark mode by default, with optional light mode if practical.

Security and Quality Requirements:

* Do not expose API keys or secrets in the frontend.
* Store sensitive settings securely on the backend.
* Validate all user inputs.
* Sanitize search/filter inputs where needed.
* Protect destructive routes with proper confirmation and authorization hooks where applicable.
* Follow clean code practices.
* Keep files organized and easy to maintain.
* Add comments where helpful, but do not over-comment obvious code.

Deliverables:

* Updated frontend templates/components/assets.
* Updated backend FastAPI routes as needed.
* Reusable UI layout and component structure.
* Modern dark glassmorphism design.
* Clean publicly available font integration.
* Beautiful rounded bubbly button system.
* Responsive mobile/tablet/desktop support.
* CRUD interface.
* Search/filter/sort/pagination tables.
* AI provider configuration with primary/backup model support.
* LangChain or modular AI abstraction integration.
* Apache ECharts dashboard components.
* Logs viewer.
* Scheduled jobs management page.
* Clear setup notes if any new dependencies are added.

Implementation Notes:

* Before making changes, inspect the existing project structure.
* Reuse existing routes, templates, static files, database models, and configuration patterns where possible.
* If the app already uses a frontend framework, continue using it unless there is a strong reason to change.
* If the app uses server-rendered templates, improve them rather than replacing the entire app unnecessarily.
* If the app already has authentication or RBAC, respect the existing system.
* If something is missing, create a clean generic implementation that can be extended later.
* Make the final UI look polished, modern, friendly, and production-ready.
