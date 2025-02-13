in @load_balanced_openai.py we are using litellm router to load-balance chat completion (and streaming) requests across a number of azure openai deployments, using a custom "low-latency" routing strategy that's implemented in @get_lowest_latency_deployment.py or @latency_routing_strategy.py

litellm is extremely bloated and unreliable and we only use azure openai anyways so there's no need for the overhead of that library. i want to do a clean-room reimplementation of this functionality, prioritizing simplicity and quality in the implementation.

let's start by thinking about the ideal API and design for a library that will solve our problem and implement that as a dedicated artifact. that library artifact should be an isolated component that we can open source. ideally the whole library will fit in a single file. here's what im thinking:

rule 0: keep it simple, avoid complexity, minimize unnecessary indirection. design for flexibility, engineer for performance

we only need to support asyncio python, and we only support azure/openai so we only have to deal with their quirks. study the implementation of the openai python library to inform your implementation decisions.

be parsimonious with taking external dependencies. off the top of my head i think we should mostly just need the openai and tenacity (for retries) libraries. we dont want to bloat our implementation

we should avoid reinventing the wheel unnecessarily

the library should be threadsafe and manage connection lifecycles/pooling correctly internally as necessary (or maybe let the underlying AsyncOpenAI/AsyncAzureOpenAI handle it if it already does so?)

the ux/devx needs to be on point. we need to think carefully about the api design to make sure we're offering the right interfaces and abstractions to users. the experience needs to be batteries-included.

the library needs to support sticky sessions, so we can optimize for prompt caching. ie we pick 1 deployment at the start of a session and try to use that deployment as much as possible for subsequent requests in the session to maximize caching. it should handle retries, timeouts, and errors correctly as well as support automatic failover to another deployment if the first one goes offline or becomes unresponsive.

its an llm loadbalancing library so we need to support all the important factors for llms like tpm/rpm ratelimits and so on. but its also a network lb library underneath so we should make sure to account for all the important factors in network lb as well.

observability needs to be unparalled and considered in every design decision. support for metrics, telemetry, and tracing needs to be top notch. some of the things we need to track/measure/expose are: input/output token counts, cached token counts and cache %, latency numbers like ttfb and total runtime, ratelimit utilization (tpm/rpm), cost, and anything else you think is valuable and high-signal for understanding llm inference server behavior. imagine you were an SRE at google or facebook and had to write this library to make your life easier as someone who spends a lot of time looking at monitoring graphs and thinking about reliability and behavior at scale.

library should use coordination-free loadbalancing (but the design should not preclude coordination). id use power of two random choices to distribute usage over all the configured deployments

library should have integral healthchecks and mark down unhealthy instances automatically. we should support the important factors involved here like cooldown on error/failure and so on. eg once a loadbalancer instance is initialized we should start a background task running periodic health checks on the configured deployments to support automatic markdown and so on

the library should have end-to-end unit and integration tests. design with testing in mind. in fact, maybe even write the tests first and the implementation itself second because that will force us to think about our desired user-facing API

we should think about the metrics strategy as well. how should we name the metrics? we'll want to support opentelemetry as well

retry and backoff functionality needs to be tuned for llm inference (streaming and non-streaming) but also carefully designed to work correctly with session stickiness and automatic failover

"prompt caching" is a feature provided by azure openai, it's not something we need to implement ourselves. we just need to manage session stickiness correctly so our cache hit rate is high. eg if we start the session by sending request 1 to azure zone A, we should send request 2 to zone A as well where the cache is hot, instead of zone B where the cache is cold. but if zone A is unhealthy/times out, then we should try zone B before failing completely

# Azure OpenAI Load Balancer Design Document

# Core Problem
Need a high-performance, reliable way to distribute LLM inference requests across multiple Azure OpenAI deployments while maximizing performance and reliability.

# Key Requirements
## Primary
Load Distribution
Power of two choices for deployment selection
Weighted by latency, health status, and current load
Support for session stickiness to maximize cache hits
Automatic failover on errors/timeouts

Health Management
Active health checking of deployments
Circuit breaker pattern for automatic markdown
Configurable cooldown periods
Automatic recovery after cooldown

Observability
OpenTelemetry integration for distributed tracing
Detailed latency metrics (TTFT, total request time)
TPM/RPM utilization tracking
Error tracking with context
Health status monitoring

Support for both Azure OpenAI and OpenAI endpoints
Configurable retry policies
Rate limit awareness (TPM/RPM)
