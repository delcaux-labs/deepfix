# DeepFix Client-Server Architecture Specifications

## Project Overview

This directory contains the complete specifications for refactoring DeepFix from a monolithic pipeline into a client-server architecture using **spec-driven development** principles.

### Vision

Transform DeepFix into a scalable, distributed system where:
- **Server**: Lightweight AI analysis service focused on intelligent diagnostics
- **Client**: Handles artifact computation, recording, and workflow integration
- **Communication**: Clean REST API with well-defined contracts

### Core Objectives

1. **Separation of Concerns**: Decouple artifact computation (client) from AI analysis (server)
2. **Scalability**: Enable independent scaling of analysis service
3. **Flexibility**: Support offline client operation with graceful degradation
4. **Maintainability**: Clear contracts between components via specifications
5. **Future-Proof**: Local-first design with cloud migration path

## Architecture Decision

**Server State Management: Hybrid (Stateless + In-Memory Cache)**

- Stateless core API for horizontal scalability
- In-memory LRU cache for KnowledgeBridge (upgradeable to Redis)
- MLflow as the persistent artifact store
- Local-first deployment model

**Key Design Choices:**

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Communication | REST API | Simple, HTTP-based, widely supported |
| Artifact Storage | Server pulls from MLflow | Simplifies client, centralizes access control |
| State Management | Stateless + cache | Enables scaling, simpler deployment |
| Deployment | Local-first | Matches current usage, easier migration |

## Specification Files

### 1. [api-spec.yaml](./api-spec.yaml)
OpenAPI 3.0 specification defining:
- REST endpoints (`/api/v1/analyze`, `/api/v1/knowledge/query`, etc.)
- Request/response schemas
- Error codes and handling
- Performance constraints

**Use this for**: API contract validation, client SDK generation, server route implementation

### 2. [service-spec.md](./service-spec.md)
Service responsibilities and constraints:
- Server responsibilities and boundaries
- Client responsibilities and boundaries
- Constraint definitions (concurrency, timeouts, resource limits)

**Use this for**: Understanding component ownership, architectural decisions

### 3. [workflow-spec.md](./workflow-spec.md)
Interaction patterns and flows:
- Primary workflow: Training → Analysis → Results
- Error scenarios and recovery strategies
- Sequence diagrams for key interactions
- Retry and timeout policies

**Use this for**: Implementation sequence, error handling logic, integration planning

### 4. [data-contracts.md](./data-contracts.md)
Data schemas and validation:
- Artifact format specifications
- Validation rules and constraints
- Backward compatibility requirements
- Migration paths for schema changes

**Use this for**: Data validation implementation, schema versioning, compatibility testing

## Implementation Phases

### Phase 1: Specification Development ✓
- [x] Define API contracts
- [x] Document service responsibilities
- [x] Specify workflows and interactions
- [x] Define data contracts

### Phase 2: Server Implementation
**Goal**: Create FastAPI server implementing the specifications

**Key Tasks:**
- [ ] Implement API endpoints per `api-spec.yaml`
- [ ] Create artifact service for MLflow integration
- [ ] Build cache service for knowledge retrieval
- [ ] Refactor orchestrator into stateless service
- [ ] Add error handling per workflow spec

**Artifacts**: `server/` directory with FastAPI application

### Phase 3: Client Implementation
**Goal**: Build Python client SDK and update integrations

**Key Tasks:**
- [ ] Create client SDK matching API spec
- [ ] Refactor Lightning callback to use client
- [ ] Implement retry logic per workflow spec
- [ ] Add offline mode fallback
- [ ] Keep artifact computation client-side

**Artifacts**: `client/` directory with SDK, updated `src/deepfix/integrations/`

### Phase 4: Deployment
**Goal**: Enable local deployment with Docker

**Key Tasks:**
- [ ] Create Dockerfile for server
- [ ] Create docker-compose for local setup
- [ ] Update pyproject.toml for client/server dependencies
- [ ] Create deployment documentation

**Artifacts**: `Dockerfile`, `docker-compose.yml`, deployment docs

### Phase 5: Documentation & Migration
**Goal**: Enable smooth transition for existing users

**Key Tasks:**
- [ ] Write architecture documentation
- [ ] Create migration guide
- [ ] Update README with new architecture
- [ ] Create backward compatibility layer

**Artifacts**: `docs/` directory with comprehensive guides

## Development Workflow

Following **spec-driven development** principles:

1. **Specifications First**: All specs are defined before implementation
2. **Contract Testing**: Validate implementations against specs
3. **Iterative Refinement**: Update specs as requirements evolve
4. **Documentation as Code**: Specs are the authoritative source

### How to Use These Specs

**For Developers:**
1. Read specifications before implementing
2. Validate code against contract definitions
3. Propose spec changes via pull requests
4. Use specs to generate tests and documentation

**For Project Managers:**
1. Track progress against implementation phases
2. Use specs to define acceptance criteria
3. Reference specs in task descriptions
4. Monitor spec compliance in reviews

**For API Consumers:**
1. Use `api-spec.yaml` for client development
2. Reference `workflow-spec.md` for integration patterns
3. Check `data-contracts.md` for data validation

## Success Criteria

- [ ] All API endpoints documented in OpenAPI spec
- [ ] Server responsibilities clearly defined and bounded
- [ ] Client-server workflows documented with error handling
- [ ] All data contracts specified with validation rules
- [ ] Server responds within 60s for typical workloads
- [ ] Client operates offline when server unavailable
- [ ] All existing agent functionality preserved
- [ ] Clear migration path for existing users

## Quick Reference

| Need | See |
|------|-----|
| API endpoint details | `api-spec.yaml` |
| Component responsibilities | `service-spec.md` |
| Request/response flow | `workflow-spec.md` |
| Data validation rules | `data-contracts.md` |
| Implementation tasks | This README (Phases 2-5) |

## Next Steps

1. **Review specifications**: Read all spec files to understand the architecture
2. **Start with server**: Implement Phase 2 following `api-spec.yaml`
3. **Build client SDK**: Implement Phase 3 following client specifications
4. **Deploy locally**: Set up development environment per Phase 4
5. **Document migration**: Create guides per Phase 5

---

**Last Updated**: October 15, 2025  
**Status**: Specifications Complete, Implementation Pending  
**Contact**: fadel.seydou@delcaux.com

