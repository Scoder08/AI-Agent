# AI-Powered Slack Server

## Overview
This project integrates an **AI-powered Slack bot** with specialized agents to review GitHub pull requests (PRs), handle order details, and generate ClickHouse SQL queries.

### Workflow
- **AI Slack Server**: Listens to Slack events (mentions & DMs) and routes queries.
- **Supervisor Agent**: Determines which specialized agent (Preview, SAM, SATWIK) should handle each request.
  - **Preview**: Reviews GitHub PR diffs for performance, bugs, and readability.
  - **SAM**: Handles operational queries related to orders and their statuses.
  - **SATWIK**: Generates optimized ClickHouse SQL queries for order analytics.

## Features
- **PR Review**: AI analyzes GitHub PR diffs and suggests optimizations.
- **Order Management**: Handles queries related to order statuses and details.
- **SQL Generation**: Generates optimized SQL queries for order analytics.

