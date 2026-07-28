# Master Prompt: Build a Modern Customer-Focused FastAPI Landing Page

You are a senior product designer, conversion-focused UX strategist, marketing copywriter, frontend engineer, security-conscious application reviewer, and Python/FastAPI developer.

Your task is to analyze this entire codebase and build a production-quality marketing landing page for the application.

The landing page must be served from the application’s root base path:

```text
/
```

The existing application must continue working correctly.

Do not remove, rename, expose, replace, or break existing:

* API routes
* User authentication
* Login pages
* Registration flows
* Static files
* Templates
* Middleware
* WebSocket endpoints
* Background tasks
* Application features
* Dashboard routes
* API integrations
* Deployment behavior

The finished landing page must be fully integrated into the existing FastAPI application and must use the repository’s existing architecture and conventions wherever practical.

---

# Primary Objective

Create an extremely modern, next-generation, dark-themed sales and marketing landing page that explains:

* What the product does
* Who the product is for
* Which customer problems it solves
* What business value it provides
* Why a prospective customer should consider purchasing it
* How a visitor can get started
* How an existing customer can log in

The page must be written primarily for:

* Prospective customers
* Buyers
* Business decision-makers
* Team leaders
* Department leaders
* Organizations
* Enterprise stakeholders
* Existing customers returning to the application

The landing page must not read like:

* Developer documentation
* An engineering project page
* An API reference
* A source-code repository
* A technical architecture overview
* A system-administration interface
* A developer setup guide
* An internal operations dashboard

The primary purpose of the page is to generate interest and encourage visitors to take a meaningful commercial action, such as:

* Purchase a plan
* Select a pricing tier
* Request a demonstration
* Contact sales
* Schedule a consultation
* Create an account
* Start a trial
* Learn more about the product

Existing customers must also have a clear Login button that takes them to the application’s existing user login page.

---

# Required Page Structure

The landing page must include:

1. Header with navigation
2. Hero section with value proposition
3. Customer problem and solution section
4. Features and benefits section
5. Product visuals, screenshots, charts, and graphs
6. How it works section
7. Customer use cases
8. Pricing section with exactly three pricing models
9. Testimonials or customer-persona section
10. FAQ section
11. Final call-to-action section
12. Footer

Additional sections may be added when supported by the codebase and useful to the sales narrative, including:

* Business outcomes
* Product demonstration
* Interactive dashboard preview
* Integrations
* Privacy and customer control
* Deployment options
* Competitive differentiation
* Industries served
* Supported workflows
* Product capabilities
* Customer success outcomes

Do not add sections simply to make the page longer.

Every section should help a prospective customer understand the product, trust it, or take the next step.

---

# Mandatory Public-Information Safety Rules

The public landing page must not expose sensitive, proprietary, private, internal, or security-relevant information.

Treat all repository content as private by default unless it is clearly intended for public marketing use.

Do not expose or include:

* Source code
* Application code snippets
* Python code
* JavaScript implementation code
* SQL queries
* Internal templates
* Internal prompts
* Proprietary algorithms
* Model-routing logic
* Internal AI instructions
* Internal business rules
* Private architecture details
* Detailed architecture diagrams
* Infrastructure diagrams
* Internal system workflows
* Internal network topology
* Internal IP addresses
* Private IP addresses
* Internal hostnames
* Server names
* Container names
* Kubernetes namespaces
* Internal ports
* Internal URLs
* Development URLs
* Staging URLs
* Private domains
* Internal API endpoints
* Administrative endpoints
* Debug endpoints
* Health endpoints that are not intended for public use
* Database connection strings
* Database credentials
* Database schemas
* Database table names
* Database column names
* Queue names
* Internal service names
* Internal storage locations
* File-system paths
* Repository paths
* Deployment paths
* Environment-variable values
* Sensitive environment-variable names
* API keys
* Access keys
* Secret keys
* Tokens
* Session tokens
* Bearer tokens
* Passwords
* Credentials
* Authentication secrets
* OAuth secrets
* Client secrets
* Encryption keys
* Signing keys
* Private certificates
* Private configuration values
* Cloud account identifiers
* Cloud credentials
* Customer data
* User data
* Personally identifiable information
* Private email addresses
* Private phone numbers
* Private account names
* Internal security configurations
* Firewall rules
* Access-control rules
* Security policies
* Vulnerability information
* Known weaknesses
* Internal logs
* Debug output
* Stack traces
* Internal monitoring data
* Operational metrics not approved for publication
* Customer usage data
* Unreleased features
* Private roadmap information
* Development notes
* Internal comments
* TODO comments containing sensitive information

Before publishing any piece of information, determine:

1. Is this useful to a prospective customer?
2. Is it safe for public disclosure?
3. Is it clearly intended for public marketing use?
4. Does it reveal how the product is internally built or operated?
5. Could it help an attacker understand the application or environment?
6. Could it reveal information about a customer, employee, developer, or internal system?

If there is any doubt, omit the information.

The repository may be used to understand the product internally, but repository content must not automatically be copied into the public website.

---

# Never Render Secrets or Private Information

Do not render secrets, configuration, private data, or internal information into:

* Visible HTML
* Hidden HTML
* HTML comments
* JavaScript
* JavaScript comments
* CSS
* CSS comments
* JSON
* JSON-LD
* Meta tags
* Open Graph metadata
* Data attributes
* Hidden inputs
* Form fields
* Browser local storage
* Browser session storage
* Client-side configuration objects
* Source maps
* Debug messages
* Error messages
* Static files
* Public network requests
* Browser console logs

Remember that hidden content, comments, minified code, browser storage, and client-side configuration are still visible through browser developer tools.

Do not place private information in the page merely because it is:

* Hidden with CSS
* Encoded
* Obfuscated
* Minified
* Stored in JavaScript
* Included inside a comment
* Added to a hidden element
* Not directly visible in the rendered interface

Do not expose server-side environment variables to templates unless each specific value is explicitly intended and approved for public display.

Never send secrets or private configuration to the browser.

---

# Customer and Sales Focus

Write all landing-page copy from the customer’s perspective.

Prioritize:

* Customer problems
* Customer outcomes
* Business value
* Time savings
* Reduced complexity
* Faster decisions
* Better visibility
* Increased control
* Improved consistency
* Improved productivity
* Easier collaboration
* Reduced manual effort
* Lower operational overhead
* Easier adoption
* Faster onboarding
* Improved privacy
* Reduced dependence on third parties
* Better reporting
* Clearer insights
* Improved decision-making

Lead with benefits before features.

Prefer customer-focused language such as:

```text
Turn complex information into clear, actionable decisions.
```

Instead of:

```text
Uses a multi-stage processing pipeline with internal services.
```

Prefer:

```text
Keep your organization’s information under your control.
```

Instead of:

```text
Runs through locally deployed containers and internal APIs.
```

Prefer:

```text
Get valuable results without rebuilding your existing workflow.
```

Instead of:

```text
Integrates through REST endpoints and asynchronous workers.
```

Prefer:

```text
See what matters, understand why it matters, and act faster.
```

Instead of:

```text
Processes records through multiple analysis components.
```

The landing page should quickly answer:

* Is this product meant for me?
* What problem will it solve?
* How will it improve my work?
* What outcome will I receive?
* Why should I trust it?
* What does it cost?
* How do I get started?
* Where do I log in?

---

# Technology Content Limits

Include very little information about the underlying technology.

Technical content should only be included when it supports:

* Buyer confidence
* Product credibility
* Security confidence
* Privacy confidence
* Deployment confidence
* Integration confidence
* Procurement decisions
* Sales qualification

Keep technical explanations:

* Brief
* Customer-friendly
* Outcome-oriented
* Easy for a nontechnical buyer to understand
* Free from internal implementation details

Acceptable examples, when supported by the codebase, include:

* Built for secure deployment
* Designed to integrate with existing workflows
* Accessible from modern web browsers
* API access is available
* Supports private deployment options
* Designed for responsive experiences
* Built to support individual users and larger organizations
* Can operate within controlled environments

Do not provide:

* Detailed technology-stack inventories
* Framework version numbers
* Dependency lists
* Library lists
* Model names unless publicly marketed
* Model inventories
* Internal AI pipeline details
* Container names
* Service-to-service communication details
* Processing algorithms
* Prompt structures
* Database details
* Internal workflows
* Deployment commands
* Developer setup instructions
* Troubleshooting instructions
* Detailed infrastructure requirements
* Internal performance tuning information
* Source-code examples
* Terminal screenshots

A prospective customer should learn what the technology enables, not how the internal system is constructed.

Use just enough technical information to make the visitor interested in purchasing, requesting a demo, or speaking with sales.

---

# Phase 1: Safely Analyze the Codebase

Before editing files, inspect the repository to understand the product.

Review, where available:

* Public-facing README content
* FastAPI application entry points
* Existing public routes
* Existing login routes
* Existing registration routes
* Existing dashboard routes
* Product features
* User workflows
* Existing public-facing terminology
* Existing templates
* Static assets
* Existing frontend components
* Approved logos
* Approved icons
* Product screenshots
* Pricing configuration
* Subscription configuration
* Licensing configuration
* Authentication entry points
* Public integrations
* Deployment options intended for customers
* Existing public marketing copy
* Tests
* Docker and deployment files
* Configuration files
* Application services
* Background jobs
* API documentation
* Data models at a conceptual level

Search the codebase for terms such as:

```text
product
features
benefits
pricing
plans
subscription
billing
license
enterprise
customer
dashboard
analytics
reports
workflow
integration
security
privacy
authentication
login
register
sign in
API
AI
automation
real-time
export
upload
processing
deployment
on-prem
cloud
documentation
```

Use private implementation details only to understand the product internally.

Do not reproduce those private implementation details on the public page.

Create an internal working summary identifying:

* Product name
* Product category
* Primary customer problem
* Primary customer outcome
* Target customer groups
* Main customer workflow
* Most valuable capabilities
* Important differentiators
* Approved integrations
* Approved deployment options
* Approved security and privacy statements
* Existing sales calls to action
* Existing registration route
* Existing login route
* Existing dashboard route
* Existing pricing information
* Existing brand colors
* Existing logos
* Existing visual assets

This internal summary must not be rendered into the public website.

---

# Accuracy and Marketing-Claim Rules

All public claims must be supported by evidence in the repository or clearly treated as editable marketing placeholders.

Do not invent:

* Customer counts
* Revenue
* Cost savings
* Productivity improvements
* Performance improvements
* Accuracy percentages
* Time-saving percentages
* Uptime guarantees
* Certifications
* Compliance claims
* Integrations
* Awards
* Testimonials
* Customer names
* Customer logos
* Pricing presented as final
* Free trials
* Cancellation guarantees
* Security guarantees
* Deployment options
* Support response times
* Service-level agreements
* Measurable customer outcomes

When information is unavailable:

1. Omit it.
2. Use restrained, factual wording.
3. Use centralized editable placeholder content.
4. Add a safe developer TODO that contains no sensitive information.

Safe example:

```html
<!-- TODO: Replace with an approved customer testimonial -->
```

Safe pricing example:

```python
# TODO: Replace initial pricing with approved production pricing.
```

Do not include:

* Internal file paths
* Repository evidence
* Internal URLs
* Security details
* Private business context

inside developer comments.

---

# FastAPI Integration

Serve the landing page from:

```python
GET /
```

Use the project’s existing FastAPI application and architecture.

Before creating or changing the route:

* Determine whether `/` already exists.
* Determine whether the project uses `APIRouter`.
* Determine whether Jinja2 templates are already configured.
* Determine whether static files are already mounted.
* Determine whether another frontend is currently served from the root.
* Determine whether middleware affects the root route.
* Determine whether authentication affects the root route.
* Determine whether the root currently redirects to another page.
* Identify the existing user login route.
* Identify the existing registration or onboarding route.
* Identify the existing authenticated-user dashboard route.

Preserve all required functionality.

Do not create a second login page.

Do not create a second authentication system.

Do not replace the existing login route.

Do not break existing redirects or session behavior.

Prefer the project’s existing route naming and URL-generation conventions.

A possible structure is:

```text
app/
├── main.py
├── routers/
│   └── landing.py
├── templates/
│   └── landing/
│       ├── index.html
│       └── partials/
└── static/
    └── landing/
        ├── css/
        │   └── landing.css
        ├── js/
        │   └── landing.js
        └── images/
```

This is only an example.

Adapt the structure to the actual repository.

Keep editable public marketing content centralized rather than duplicating it throughout the HTML.

A server-side data structure may contain:

* Product content
* Feature descriptions
* Pricing plans
* FAQs
* Testimonials
* Use cases
* CTA labels
* Public routes

Do not put secret or private values in this structure.

---

# Existing Login Button Requirement

Include a clearly visible Login button in both:

* Desktop navigation
* Mobile navigation

The Login button must take the user to the application’s existing user login page.

Before implementing it:

* Inspect the FastAPI routes and templates.
* Identify the existing login route.
* Identify its route name when available.
* Use the project’s existing URL-generation mechanism.
* Preserve any application base path or URL prefix.
* Preserve any reverse-proxy path handling.
* Preserve existing authentication redirects.
* Do not hard-code `/login` unless it is confirmed to be the correct route.
* Do not create a duplicate login page.
* Do not link to an administrative login page.
* Do not link to a developer authentication endpoint.
* Do not link to an internal identity-provider endpoint unless that is already the intended public login flow.
* Do not expose authentication configuration or identity-provider details.

When FastAPI or Starlette named routes are available, prefer safe route generation, such as:

```jinja2
{{ request.url_for("login") }}
```

Adapt the route name to the actual application.

The header should distinguish between:

* Login: secondary action for existing customers
* Get Started, View Pricing, Request a Demo, or Contact Sales: primary sales action for prospective customers

The Login button should:

* Be easy to find
* Be visually secondary to the main sales CTA
* Use a polished outline, translucent, or glass-style design
* Include a clear accessible label
* Include a visible keyboard-focus state
* Work in desktop navigation
* Work in mobile navigation
* Navigate directly to the existing user login flow
* Respect the application’s configured base path

If the application already supports authentication-state detection, then authenticated users may see an appropriate existing-user action such as:

* Open Dashboard
* My Account
* Continue to Application

Do not invent authentication-state behavior if it is not already supported by the application.

Verify that:

* The Login button points to the correct route.
* The destination returns successfully.
* Authentication redirects still work.
* Session handling remains intact.
* The button respects deployment prefixes.
* No authentication secrets appear in page source.
* No credentials appear in browser code.
* No private identity-provider information is exposed.

---

# Design Direction

Create an extremely modern, premium dark-theme experience with tasteful glassmorphism.

The visual direction should include:

* Deep charcoal backgrounds
* Near-black backgrounds
* Midnight-blue backgrounds
* Layered radial gradients
* Subtle mesh gradients
* Subtle grid patterns
* Soft noise textures
* Optional lightweight particle effects
* Frosted translucent panels
* Soft low-opacity borders
* Controlled background blur
* Refined shadows
* Gradient accents
* Large confident typography
* Strong visual hierarchy
* Premium spacing
* Rounded cards
* Subtle depth
* Smooth transitions
* Responsive layouts
* High-quality product visuals
* Refined micro-interactions

The result should feel like a premium technology company’s sales website.

It should not feel like:

* An admin dashboard
* A generic Bootstrap template
* A developer portfolio
* A technical documentation site
* A copied marketplace theme
* A collection of identical cards

Use glassmorphism selectively so text remains readable.

Example treatment:

```css
.glass-panel {
    background: rgba(255, 255, 255, 0.055);
    border: 1px solid rgba(255, 255, 255, 0.10);
    box-shadow:
        0 24px 80px rgba(0, 0, 0, 0.30),
        inset 0 1px 0 rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}
```

Include a graceful fallback for browsers that do not support backdrop filters.

Avoid:

* Excessive neon
* Excessive motion
* Constant animations
* Low-contrast text
* Excessive blur
* Random gradients
* Generic stock-template layouts
* Repetitive feature cards
* Fake command-line windows
* Source-code displays
* Terminal screenshots
* Technical architecture diagrams
* Internal workflow diagrams
* Decorative charts that communicate nothing
* Large background videos that hurt performance

---

# Design Inspiration

Use these websites for broad visual inspiration only:

* `https://devport-live-demo-beta.vercel.app/`
* `https://themes.incognitothemes.com/spyder/index.html`
* `https://templates.codeinsolution.com/html/syniq/`

Draw inspiration from:

* Premium single-page storytelling
* Large product-focused hero sections
* Smooth anchored navigation
* Glassmorphism
* Gradient lighting
* Dashboard-style product previews
* Interactive data visuals
* Metric cards
* Alternating feature layouts
* Clear pricing presentation
* Strong calls to action
* Refined FAQ sections
* Modern responsive behavior
* Subtle parallax
* Layered motion
* Product-led storytelling

Do not copy:

* Their source code
* Their wording
* Their branding
* Their logos
* Their assets
* Their exact layout
* Their color palette
* Their illustrations

The final design must be visually original and tailored to this product.

---

# Header and Navigation

Create a polished sticky or floating header.

Include:

* Product logo or wordmark
* Product name
* Navigation links
* Login button
* Primary sales CTA
* Mobile navigation menu

Suggested navigation:

```text
Product
Benefits
How It Works
Pricing
Customers
FAQ
```

Only include destinations that exist on the page or valid application routes.

The header should:

* Use a translucent or glass-style appearance
* Have strong text contrast
* Change subtly after scrolling
* Support smooth anchor scrolling
* Include visible keyboard-focus states
* Be fully usable on mobile
* Avoid horizontal overflow
* Use proper accessible menu controls
* Use `aria-expanded` where appropriate
* Close the mobile menu after navigation
* Prevent background scroll when appropriate

Possible primary sales CTAs include:

* Get Started
* Start Your Trial
* Request a Demo
* Schedule a Consultation
* Contact Sales
* View Plans

Do not create dead links.

Do not use placeholder `href="#"` links.

All links must point to:

* Real page sections
* Existing application routes
* Approved external destinations

---

# Hero Section

The hero must communicate the product’s value immediately.

Include:

* Optional category or announcement badge
* Strong benefit-focused headline
* Concise value proposition
* Primary sales CTA
* Secondary CTA
* Product visual
* Optional verified trust indicators
* Login access in the main navigation

The headline should communicate a customer outcome, not a technical capability.

The supporting text should briefly explain:

* The customer’s problem
* How the product helps
* Who benefits
* Why the product is worth exploring

Avoid technical jargon.

The hero should be understandable within a few seconds.

Suggested structure:

```text
Eyebrow:
A short product category or differentiator

Headline:
A high-impact statement describing the primary customer result

Supporting copy:
Two or three sentences describing the problem, solution, and audience

Primary CTA:
Get Started, Request a Demo, View Pricing, or Contact Sales

Secondary CTA:
Explore Features, See How It Works, or View Product
```

Connect CTAs only to real routes or page sections.

---

# Hero Product Visual

Create a visually rich product preview based on real application capabilities.

Possible visual components include:

* Dashboard preview
* Business analytics panel
* Workflow progress display
* Report preview
* Insights panel
* Recommendation panel
* Timeline
* Activity overview
* Status summary
* Collaboration workspace
* Processing overview
* Trend chart
* Product interface mockup
* Customer outcome summary

Whenever practical, use approved screenshots already available in the repository.

If screenshots do not exist, construct polished HTML, CSS, SVG, or chart-based mockups based on real product concepts.

Do not present mock functionality as real functionality.

Use safe sample labels and data.

Clearly label illustrative content where appropriate:

```text
Illustrative dashboard
Example report
Sample workflow
Demo data
Product preview
```

Do not include:

* Internal logs
* Real customer data
* Real user data
* Email addresses
* IP addresses
* Private URLs
* Source code
* Terminal output
* Internal model names
* Internal API names
* System architecture
* Infrastructure details
* Security event details
* Debug information

Add restrained motion such as:

* Slowly shifting gradients
* Floating status cards
* Animated chart paths
* Small value transitions
* Staggered card entrances
* A subtle live indicator
* Gentle parallax

Respect `prefers-reduced-motion`.

---

# Customer Problem and Solution Section

Create a clear section explaining:

* The customer’s current pain
* Why the current process is inefficient or difficult
* How the product improves the situation
* What the customer receives

Use concise, customer-focused language.

A useful structure may be:

```text
The problem:
Customers struggle with fragmented tools, manual work, slow decisions, or poor visibility.

The solution:
The product brings the important information together and turns it into clear actions.

The outcome:
Customers save time, gain clarity, and make better-informed decisions.
```

Replace these generic concepts with the real product value found in the repository.

Do not describe internal implementation.

---

# Features and Benefits

Build this section around the strongest customer-facing capabilities discovered in the codebase.

Each feature should explain:

* The customer challenge
* The product capability
* The resulting benefit
* Why the benefit matters

Lead with benefit-oriented titles.

Prefer:

```text
Find what matters faster
```

Instead of:

```text
Multi-stage analysis engine
```

Prefer:

```text
Keep everyone working from the same information
```

Instead of:

```text
Shared relational data layer
```

Prefer:

```text
Turn results into clear next steps
```

Instead of:

```text
Structured response generation
```

Use varied visual layouts, including:

* Bento grids
* Large featured capabilities
* Alternating image-and-copy sections
* Outcome cards
* Product-preview panels
* Compact supporting features

Avoid presenting every feature as an identical icon card.

Each major feature may include:

* Icon
* Short benefit-oriented title
* Concise customer-focused description
* Product screenshot or mockup
* Graph or visualization
* Small capability badges

Do not expose internal file paths or technical implementation evidence on the page.

---

# Product Visuals, Charts, and Graphs

Use product visuals and graphs throughout the page where they improve the customer’s understanding.

Suitable visualization types include:

* Time-series line chart
* Area chart
* Bar chart
* Donut chart
* Gauge
* Heatmap
* Timeline
* Workflow diagram
* Usage distribution
* Performance summary
* Status matrix
* Geographic view
* Progress visualization
* Before-and-after comparison
* Customer outcome summary

Choose visualizations that fit the product.

Do not add meaningless charts solely for decoration.

Use real product concepts but safe illustrative data.

Charts may communicate:

* Work completed
* Time saved conceptually
* Trends identified
* Tasks organized
* Insights generated
* Workflow progress
* Usage over time
* Reporting categories
* Operational visibility
* Product adoption

Do not present illustrative values as verified customer results.

When using demonstration data, clearly label it:

```text
Example workflow
Sample analysis
Illustrative dashboard
Demo dataset
```

Do not chart:

* Real customer data
* Private system metrics
* Internal infrastructure metrics
* Internal model performance
* Security vulnerabilities
* Sensitive operational data
* Unapproved business statistics

Reuse the project’s existing charting library when practical.

Otherwise, use one lightweight, permissively licensed library such as:

* Apache ECharts
* Chart.js
* D3.js for genuinely custom visualization needs

Keep the number of frontend libraries small.

Chart requirements:

* Responsive
* Dark-theme compatible
* Accessible summary text
* Clear labels
* Clear units
* Proper legends
* No fabricated claims
* Graceful behavior if JavaScript is unavailable

---

# How It Works

Explain the customer journey in three to five simple steps.

A generic example is:

```text
01 — Add or Connect Your Information
02 — Let the Product Organize and Analyze It
03 — Review Clear Results
04 — Take Action
```

Replace this with the product’s actual customer workflow.

Focus on:

* What the customer does
* What the customer sees
* What the customer receives
* What action the customer can take

Do not explain:

* Internal processing stages
* Internal services
* Queues
* Databases
* Models
* Algorithms
* Prompts
* Network communication
* Infrastructure

A customer should understand how easy the product is to use without learning how the backend works.

---

# Customer Use Cases

Show how the product supports its most important customer groups.

For each use case, explain:

* Who the customer is
* What challenge they face
* How the product fits into their work
* What result they receive

Potential audiences may include:

* Individuals
* Professionals
* Power users
* Small teams
* Businesses
* Organizations
* Analysts
* Operations teams
* Security teams
* Financial teams
* Educational users
* Enterprise customers

Only include audiences supported by the repository.

Do not invent industries or use cases that the product does not support.

---

# Privacy, Security, and Customer Control

When supported by the repository, include a concise customer-facing section about privacy, security, or deployment control.

Emphasize customer benefits such as:

* Greater control over information
* Flexible deployment options
* Secure account access
* Reduced unnecessary third-party exposure
* Support for organizational requirements
* Controlled access
* Private deployment
* Customer-owned environments

Do not reveal how security controls are implemented.

Do not publish:

* Authentication architecture
* Encryption implementation details
* Firewall rules
* Internal access-control rules
* Security configurations
* Network design
* Vulnerability details
* Security testing details
* Penetration-test results
* Incident information
* Authentication providers unless publicly approved
* Internal security products
* Secret-management systems

Do not claim compliance with:

* SOC 2
* HIPAA
* PCI DSS
* FIPS
* FedRAMP
* GDPR
* ISO 27001
* CMMC
* Any other standard

unless explicit approved evidence exists.

---

# Pricing Section

Create a prominent pricing section containing exactly three pricing models.

These plans are initial placeholders and will be adjusted later.

Store all pricing content in one centralized, easily editable data structure.

Add a safe source-code comment:

```python
# TODO: Replace initial pricing with approved production pricing.
```

Do not scatter pricing values throughout multiple templates or JavaScript files.

The pricing section should make plan comparison easy.

Each pricing card must include:

* Plan name
* Price
* Billing period
* Intended customer
* Short description
* Included benefits
* CTA
* Featured-plan indicator where applicable
* Clear visual differentiation

Use the following initial pricing models.

## Plan 1: Individual

Designed for a single user who wants access to the core product experience.

Initial placeholder price:

```text
$19 per month
```

Suggested benefits:

* Core product features
* Individual workspace
* Standard usage limits
* Basic reports or exports
* Standard support

Suggested CTA:

```text
Get Started
```

## Plan 2: Professional

Designed for power users, professionals, or small teams requiring higher usage levels and additional capabilities.

Initial placeholder price:

```text
$49 per month
```

Mark this plan as:

```text
Most Popular
```

Suggested benefits:

* Everything in Individual
* Higher usage limits
* Advanced product capabilities
* Expanded reports or analytics
* Priority support
* Collaboration features when supported

Suggested CTA:

```text
Choose Professional
```

## Plan 3: Enterprise

Designed for businesses and organizations requiring larger deployments, custom requirements, sales-assisted onboarding, or tailored licensing.

Price:

```text
Custom pricing
```

Suggested benefits:

* Everything in Professional
* Organization-wide deployment
* Custom usage levels
* Administrative capabilities when supported
* Deployment planning
* Priority onboarding
* Contract and licensing options
* Dedicated support options when available

Suggested CTA:

```text
Contact Sales
```

These prices and benefits are editable placeholders.

Do not imply that they are final, approved, or currently available unless confirmed.

Do not prominently display the word “placeholder” in the customer-facing design.

Do not include:

* Hidden fees
* Unsupported features
* Invented trial periods
* Invented refund policies
* Invented cancellation guarantees
* Invented service-level agreements
* Unsupported support promises
* Unsupported collaboration features
* Unsupported administrative functionality

If a listed suggested benefit is not supported by the codebase, remove it or replace it with an accurate customer-facing benefit.

Pricing CTA buttons must link to real routes or approved destinations.

---

# Testimonials Section

Use real testimonials only when approved testimonials exist in the repository or approved public content.

Do not fabricate:

* Customer names
* Company names
* Quotes
* Job titles
* Photos
* Customer logos
* Measurable outcomes
* Ratings
* Review counts

When verified testimonials are unavailable, use a safe alternative such as:

* Customer personas
* Example customer scenarios
* Common customer outcomes
* “Built for teams that…” cards
* Industry use cases
* Clearly marked draft testimonial data that is not rendered publicly

The source structure should make it easy to add approved testimonials later.

Do not display fake endorsements as real customer feedback.

---

# FAQ Section

Create six to ten useful customer-focused questions.

Potential topics include:

* What does the product do?
* Who is it designed for?
* How quickly can I get started?
* Do I need technical experience?
* What pricing plans are available?
* Can businesses purchase it?
* Can it fit into existing workflows?
* What deployment options are available?
* How is customer information handled?
* How do I request a demonstration?
* Is support available?
* Can the product scale for larger organizations?
* Where do existing customers log in?

Keep answers:

* Clear
* Concise
* Customer-focused
* Sales-oriented
* Accurate
* Free from internal technical details

Use an accessible accordion.

Requirements:

* Keyboard operable
* Proper buttons
* Correct `aria-expanded`
* Visible focus states
* Server-rendered FAQ text
* Usable without animation
* Compatible with reduced-motion preferences

---

# Calls to Action

Use clear and consistent calls to action throughout the page.

Choose one primary conversion action and repeat it strategically.

Possible primary actions:

* Get Started
* Start Your Trial
* Request a Demo
* Schedule a Consultation
* Contact Sales
* View Pricing

Use a secondary action for visitors who need more information:

* Explore Features
* See How It Works
* View Product
* Read the FAQ

Keep Login visually distinct as the existing-customer action.

Connect buttons only to:

* Real application routes
* Real page sections
* Approved external destinations

Do not use dead links.

Do not use placeholder links.

---

# Final Call-to-Action Section

Before the footer, add a strong conversion-focused section.

Include:

* Outcome-focused headline
* Brief supporting text
* Primary CTA
* Optional secondary CTA
* Clear visual distinction

Example purpose:

```text
Ready to turn complex work into clearer decisions?
```

Replace generic wording with product-specific messaging.

Connect the CTA to a real route, such as:

* Registration
* Demo request
* Contact sales
* Pricing
* Existing onboarding flow

---

# Footer

Include appropriate groups such as:

* Product
* Pricing
* Resources
* Company
* Legal

Possible links include:

* Features
* Benefits
* Pricing
* FAQ
* Login
* Contact
* Privacy Policy
* Terms
* Support
* Documentation when customer appropriate

Only include real links.

Do not expose:

* Administrative links
* Internal documentation
* Internal API documentation
* Debug pages
* Private status pages
* Private repositories
* Development resources
* Infrastructure pages
* Monitoring pages

Also include:

* Product name
* Short product description
* Copyright notice
* Dynamically generated current year
* Approved social links when available
* Approved legal links

---

# Visual Assets

Use existing approved assets whenever available:

* Product screenshots
* Logos
* Icons
* Illustrations
* Backgrounds
* Diagrams
* Interface captures
* Brand marks

Review every asset for private or sensitive information before displaying it.

Do not use screenshots containing:

* Customer names
* Email addresses
* Private messages
* API keys
* Tokens
* Internal URLs
* IP addresses
* Internal hostnames
* Debug output
* Source code
* Logs
* Private files
* Sensitive account information

Redact or do not use unsafe screenshots.

Optimize images:

* Prefer WebP or AVIF where practical
* Preserve high-quality sources
* Use responsive image sizes
* Add descriptive alt text
* Lazy-load below-the-fold images
* Set width and height to prevent layout shift

Do not hotlink unreliable third-party images in production.

When the codebase lacks suitable images, create product-oriented visuals using:

* HTML
* CSS
* SVG
* Safe illustrative data
* Existing icon systems
* Existing charting libraries

Avoid generic stock photography unless highly relevant, approved, and legally suitable.

---

# Typography and Icons

Use a modern typography system.

Prefer the project’s existing font.

Otherwise use a reliable system or open-source font stack, such as:

```css
font-family:
    Inter,
    Manrope,
    "SF Pro Display",
    "Segoe UI",
    system-ui,
    sans-serif;
```

Use one consistent icon system.

Prefer an icon library already installed.

Suitable lightweight options include:

* Lucide
* Heroicons
* Phosphor Icons

Do not mix multiple unrelated icon systems.

---

# Animation and Interaction

Use tasteful interaction design, including:

* Fade and slide entrances
* Staggered reveals
* Header background transitions
* Smooth anchor scrolling
* Hover elevation
* Border-light movement
* Subtle card tilt
* Animated product workflow
* Chart entrance animation
* Accordion transitions
* Mobile-menu animation
* Verified metric count-ups when appropriate

Animation must remain secondary to the content.

Avoid:

* Scroll-jacking
* Excessive parallax
* Constant motion
* Fast flashing
* Distracting glow effects
* Essential information that only appears on hover

Support reduced-motion preferences:

```css
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        scroll-behavior: auto !important;
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
```

---

# Responsive Design

The page must work well at:

* 320px mobile width
* 375px phones
* 390px phones
* 430px phones
* Tablets
* Standard laptops
* Desktop monitors
* Ultra-wide displays

Test:

* Navigation wrapping
* Mobile navigation
* Login-button visibility
* Hero overflow
* Long headings
* Product-preview scaling
* Chart resizing
* Pricing-card stacking
* FAQ width
* Footer columns
* Touch-target size
* Landscape mobile layouts
* Browser zoom
* Long localized text where practical

Do not merely shrink the desktop layout.

Recompose sections intentionally for mobile.

---

# Accessibility

Target WCAG 2.2 AA practices.

Include:

* Semantic HTML landmarks
* Correct heading hierarchy
* Exactly one clear `<h1>`
* Skip-to-content link
* Accessible navigation
* Keyboard-operable components
* Visible focus indicators
* Sufficient color contrast
* Descriptive alternative text
* Accessible accordion behavior
* Accessible mobile-menu behavior
* Reduced-motion support
* Meaningful link text
* Non-color status indicators
* Proper button semantics
* Accessible chart summaries
* Accessible Login button

Decorative visuals should use:

```html
aria-hidden="true"
```

Do not use placeholder text as the only form label.

Do not rely on color alone to communicate meaning.

---

# SEO and Social Metadata

Add appropriate metadata derived from the product:

* Page title
* Meta description
* Canonical URL when known
* Open Graph title
* Open Graph description
* Open Graph image
* Twitter/X card metadata
* Theme color
* Favicon links
* Robots directives
* Structured data when justified

Use valid JSON-LD where appropriate, such as:

* `SoftwareApplication`
* `Organization`
* `WebSite`
* `FAQPage`

Do not include:

* Unsupported review ratings
* Fake review counts
* Unsupported pricing
* Private URLs
* Internal product names
* Internal architecture details
* Private contact information
* Internal IDs

Ensure the page contains exactly one clear `<h1>`.

---

# Performance

Aim for excellent Core Web Vitals.

Requirements:

* Avoid large render-blocking dependencies
* Defer nonessential JavaScript
* Minimize layout shift
* Optimize image delivery
* Lazy-load below-the-fold assets
* Avoid oversized videos
* Reduce unused CSS
* Use efficient animations
* Avoid excessive DOM complexity
* Limit third-party scripts
* Cache static assets appropriately
* Avoid loading multiple chart libraries
* Avoid unnecessary web fonts
* Prevent hidden mobile content from loading large unnecessary assets

The page should feel visually rich without becoming slow.

Do not sacrifice loading performance for decorative effects.

---

# Security Requirements

Do not introduce unsafe frontend or backend behavior.

Requirements:

* Do not expose secrets
* Do not expose environment variables
* Do not embed API keys
* Do not embed private tokens
* Do not disable existing security middleware
* Do not weaken authentication
* Do not bypass authorization
* Do not inject untrusted HTML
* Escape dynamic template content
* Preserve CSRF protections where applicable
* Validate form input
* Avoid unsafe `innerHTML`
* Use safe URL generation
* Avoid unnecessary third-party scripts
* Preserve secure cookies
* Preserve authentication redirects
* Preserve existing session behavior
* Do not expose private internal routes
* Do not expose private hostnames
* Do not expose infrastructure details
* Do not expose source maps containing internal source information in production

If a contact, demo-request, or newsletter form is added:

* Connect it only to a secure existing backend flow.
* Validate all input.
* Apply existing abuse protections.
* Do not invent a fake submission endpoint.
* Do not send form content to an unapproved third party.
* Do not include private recipient addresses in browser code.

If no secure backend flow exists, use a real approved contact route or omit the form.

---

# Dependency Strategy

Prefer the existing frontend stack.

If the repository already uses:

* Tailwind CSS
* Bootstrap
* Alpine.js
* HTMX
* React
* Vue
* Svelte
* Vanilla JavaScript
* A component library
* A charting library

reuse it where practical.

Do not introduce a heavy frontend framework solely for the landing page unless the project already uses it or there is a strong architectural reason.

Prefer:

* Semantic HTML5
* Modern CSS
* Lightweight JavaScript
* Progressive enhancement
* Minimal runtime dependencies
* Locally hosted assets where practical

Avoid introducing a complex Node.js build pipeline into a server-rendered FastAPI project unless one already exists.

---

# Code Quality

Produce maintainable code.

Requirements:

* Follow repository formatting conventions
* Follow repository naming conventions
* Keep marketing content centralized
* Keep pricing centralized
* Keep FAQs centralized
* Keep testimonials centralized
* Separate structure, styling, and behavior
* Avoid giant inline CSS blocks
* Avoid giant inline JavaScript blocks
* Use reusable template partials when helpful
* Avoid duplicated content
* Preserve type hints
* Use safe URL generation
* Use named FastAPI routes when available
* Keep changes focused on the landing page
* Do not expose private internal names in public variable names sent to the browser
* Do not leave debug code
* Do not leave console logging
* Do not leave unused assets

Possible template partials include:

```text
_header.html
_hero.html
_feature_card.html
_pricing_card.html
_testimonial_card.html
_faq_item.html
_cta.html
_footer.html
```

Do not over-engineer a small project.

---

# Testing and Validation

After implementation, verify all of the following.

## Backend

* Application starts successfully.
* `GET /` returns HTTP 200.
* The root response is HTML.
* Existing routes still work.
* Existing API routes still work.
* Existing authentication still works.
* Existing login route still works.
* Static assets return successfully.
* Template paths resolve correctly.
* There are no import errors.
* There are no route conflicts.
* There are no new unhandled exceptions.
* Existing middleware remains active.
* Existing redirects remain correct.

## Login

* The Login button points to the existing login page.
* The Login button works in desktop navigation.
* The Login button works in mobile navigation.
* The Login destination returns successfully.
* Authentication redirects remain intact.
* Session handling remains intact.
* The Login route respects the application base path.
* No authentication configuration is exposed.
* No credentials are exposed.
* No identity-provider secrets are exposed.

## Frontend

* No browser-console errors
* Navigation links work
* Smooth scrolling works
* Mobile navigation works
* Mobile navigation is accessible
* Login button works
* FAQ accordion works
* Pricing cards display correctly
* Charts render correctly
* Animations respect reduced-motion settings
* CTA links point to real destinations
* Images load
* No broken links
* No horizontal scrolling
* Forms behave safely
* Layout works on mobile and desktop

## Information Disclosure

Review the final rendered page, HTML source, JavaScript, CSS, metadata, network requests, and static assets.

Confirm that none of the following are exposed:

* Code
* Secrets
* API keys
* Passwords
* Tokens
* Private internal information
* Internal URLs
* Internal IP addresses
* Hostnames
* Ports
* File paths
* Repository paths
* Architecture details
* Database details
* Internal service names
* Customer data
* User data
* Private configuration
* Debug information
* Stack traces
* Internal logs
* Private business information

## Quality

* Content accurately represents the product.
* Copy is focused on customers and sales.
* Technical content is minimal.
* No unsupported marketing claims are present.
* No lorem ipsum remains.
* No fake testimonials appear as real.
* No fabricated customer metrics appear.
* No placeholder links remain.
* Product terminology is consistent.
* Pricing is stored centrally.
* Pricing is clearly editable in source.
* The page is responsive.
* The page is accessible.
* The design looks intentional on desktop and mobile.
* The site feels like a premium commercial product.

Run the project’s existing tests, linting, and formatting tools.

Add a focused test for the root route if the project already has an established testing framework.

Example:

```python
def test_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
```

Adapt the test to the repository’s actual structure.

Do not claim any command or test passed unless it was actually executed successfully.

---

# Implementation Process

Follow this process:

1. Inspect the repository.
2. Identify the FastAPI application architecture.
3. Identify the product’s actual customer-facing value.
4. Identify the target customers.
5. Identify the existing root route.
6. Identify the existing login route.
7. Identify registration, dashboard, and contact routes.
8. Identify templates and static-file configuration.
9. Identify reusable approved assets.
10. Review assets for private or sensitive information.
11. Identify existing frontend and charting libraries.
12. Create an internal customer-focused content map.
13. Separate public marketing facts from private implementation details.
14. Design the landing-page information architecture.
15. Implement the root FastAPI route.
16. Implement the existing-login-route integration.
17. Implement templates and reusable components.
18. Implement centralized pricing data.
19. Implement styling.
20. Implement lightweight interactions.
21. Add safe product visuals and graphs.
22. Connect CTAs to real routes.
23. Add SEO and accessibility metadata.
24. Test responsive behavior.
25. Test the existing login flow.
26. Test existing application functionality.
27. Run available tests and linting.
28. Review every public claim against repository evidence.
29. Inspect the rendered page for information disclosure.
30. Remove temporary content, debug code, and unsafe comments.
31. Summarize the completed work.

Do not stop after creating a design proposal.

Implement the complete working landing page.

---

# Required Final Response

After making the changes, provide the following.

## Product Findings

Summarize:

* What the application does
* Primary target customers
* Main customer problem
* Main value proposition
* Most important customer-facing capabilities
* Any missing marketing information

Do not include private implementation details in this summary.

## Files Changed

List every file:

```text
Created:
- path/to/file

Modified:
- path/to/file
```

## Implementation Summary

Explain:

* How `/` is served
* How the existing Login route was identified and connected
* How static assets are mounted
* Which templates and components were created
* Which visualization library was used
* How responsive behavior was handled
* How accessibility was handled
* How pricing is centralized
* How public content was separated from private implementation details

## Verification

Report:

* Commands executed
* Tests run
* Linting performed
* Formatting performed
* Application startup result
* Root route result
* Login route result
* Browser or rendering checks
* Responsive checks
* Information-disclosure review

## Outstanding Placeholders

Explicitly list anything requiring owner approval or replacement, including:

* Final pricing
* Approved testimonials
* Customer logos
* Legal links
* Contact details
* Production domain
* Analytics identifiers
* Approved performance statistics
* Social accounts
* Trial terms
* Support commitments
* Enterprise licensing details

Do not claim a command or test passed unless it was actually executed successfully.

---

# Final Quality Bar

The finished page should look like a polished, commercially credible website suitable for a premium:

* AI product
* Analytics platform
* Cybersecurity product
* Automation platform
* Financial application
* Productivity application
* Business SaaS application
* Enterprise software product

The page must be:

* Visually impressive
* Product-specific
* Customer-focused
* Sales-oriented
* Informative
* Trustworthy
* Responsive
* Accessible
* Fast
* Maintainable
* Secure
* Correctly integrated with FastAPI

The result must not look like:

* A generic Bootstrap template
* An admin dashboard pasted onto a homepage
* A developer documentation page
* A technical architecture overview
* A collection of identical feature cards
* An unfinished template
* A page filled with fictional metrics
* A copied version of an inspiration website
* A page that exposes source code or internal system details

Use the repository as the source of truth.

Use private code only to understand the product.

Expose only safe, customer-relevant, approved information.

Focus the final experience on helping prospective customers understand the value, compare the three pricing plans, take a sales action, and access the existing user login page.
