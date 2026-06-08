# By: Christopher Gray  |  Version: 0.0.21  |  Updated: 6/8/2026  | github.com/c2theg/ai
#-----------------------------------------------------------------------------------------

## Rules

- Always ask clarifying questions before starting a complex task or any task involving modifing a database
- Show your plan and steps before executing
- Keep reports and summaries concise — bullet points over paragraphs.  use as less tokens as possible
- Save all output files to the `output/` folder
- Cite sources when doing research
- Provide Pro's and Con's when 2+ options are available
- Do not ask for permission to repeat the same class of task more than once
- Security is first priority — RBAC must be enforced on every page
- Always generate a `readme.md` for deliverables (what it does, architecture/flow diagrams, intended audience, how to use, security risks + mitigations)
- Provide a method to rollback changes if needed.
- Backup code before making changes. If changes are successful, delete the backup.
- Automatically test code after making changes
- Provide a visual diff of changes made
- Provide a visual work flow on how to test changes (where possible)
- When creating containers, always choose the latest version of the technology that supports all the features, and is stable (not beta / alpha). (ie: python3.13+ where possible)
- Come up with a token count for the desired task before starting. Tell me what that count is, and if I want to proceed. Also gather remaining available tokens and let me know if I have enough to complete the task, before starting.
- When creating public documentation do the following:
    - Can you take the '<README.MD FILE HERE>' file and create a beautiful, clean, marketing friendly, easy to read documentation page on /docs.php. and give it the title "<README.MD FILE TITLE>" at the top navigation
    - Tone throughout: Confident, forward-thinking, customer-benefit focused. No code, no JSON, no implementation details, no proprietary specifics.
    - this should be human readable to a new or existing customer, not a developer. this is external facing so dont put anything properitary
    - dont include any json or code examples. this is not for someone to impliment their own solution. just convay how we built the best solution and they should buy ours
- do not allow for api keys to be exposed in the code or documentation
- make sure to always use the strongest security measures and best practices
- I want you to challenge me. Dont just do anything i say without deciding if its the best approach
- If you don't understand something, dont hesitate to ask me instead of making assumptions
- Before compaction, save all context to the database.
- After compaction, restore relevant context.
- If the user is frustrated, stop iterating on the current approach.
- Search the brain for what was discussed and reassess.

- Make sure to update the date & time and version number of all files when making changes.  Append an "Updated by: AI (<Enter AI Name>)" comment, below the Updated value
- include a very brief list of bullet points bellow that for all changes made, and include a DateTime & Version & "AI - <Enter AI Name>", at the end of each line. Only allow the most 10 recent changes, and if more than 10, delete the oldest one.
- When I hit my limit, issue the following command to identify what account is currently signed in: 'claude auth status'

- Keep a running list of changes (change log) in a file called 'changes_customers.md' with all changes that have been done to the website and application, by calendar quarter
- At the end of each quarter, create a customer-facing summary of major changes call: 'changes_customers_[Q1|Q2|Q3|Q4]_[YEAR].md' in the root directory of the application
    - Make these short (less then 30 word each) bulletpoints
    - Order them by importance
    - The audiance are customers, so it should not include code, jargon, architecture, or anything technical. Focus on benefits and improvements.
    - Dont mention issues, bugs, basic code iterations, etc. Only mention successful changes
    - Provide no more then 8 entries / bulletpoints
- At the end of each month, create a customer-facing full summary of major changes call: 'changes_customers_[Q1|Q2|Q3|Q4]_[YEAR]_full.md' in the root directory of the application.
    - this should be used as a blog like format, which provides longer, more rich details about the change, and not limited to length
    - Order them by importance
    - The audiance are customers, so it should not include code, jargon, architecture, or anything technical. Focus on benefits and improvements.
    - Dont mention issues, bugs, basic code iterations, etc. Only mention successful changes
- At the end of each month, create a architecture-facing full summary of all changes call: 'changes_architect_[Q1|Q2|Q3|Q4]_[YEAR]_full.md' in the root directory of the application.
    - this should be used as a blog like format, which provides longer, more rich details about the change, and not limited to length
    - Order them by importance
    - The audiance are developers and architectes, so it should not include detailed code (only high level).
    - Include benefits and improvements for the change

