package extproc

import (
	"fmt"
	"os"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	candle_binding "github.com/aphrodite-engine/aphrodite-engine/aphrodite_router/candle_binding"
	"github.com/aphrodite-engine/aphrodite-engine/aphrodite_router/router/pkg/cache"
	"github.com/aphrodite-engine/aphrodite-engine/aphrodite_router/router/pkg/classification"
	"github.com/aphrodite-engine/aphrodite-engine/aphrodite_router/router/pkg/config"
	"github.com/aphrodite-engine/aphrodite-engine/aphrodite_router/router/pkg/observability/logging"
	"github.com/aphrodite-engine/aphrodite-engine/aphrodite_router/router/pkg/services"
	"github.com/aphrodite-engine/aphrodite-engine/aphrodite_router/router/pkg/tools"
	"github.com/aphrodite-engine/aphrodite-engine/aphrodite_router/router/pkg/utils/pii"
)

// OpenAIRouter is an Envoy ExtProc server that routes OpenAI API requests
type OpenAIRouter struct {
	Config               *config.RouterConfig
	CategoryDescriptions []string
	Classifier           *classification.Classifier
	PIIChecker           *pii.PolicyChecker
	Cache                cache.CacheBackend
	ToolsDatabase        *tools.ToolsDatabase
}

// Ensure OpenAIRouter implements the ext_proc calls
var _ ext_proc.ExternalProcessorServer = (*OpenAIRouter)(nil)

// NewOpenAIRouter creates a new OpenAI API router instance
func NewOpenAIRouter(configPath string) (*OpenAIRouter, error) {
	// Always parse fresh config for router construction (supports live reload)
	cfg, err := config.Parse(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}
	// Update global config reference for packages that rely on config.GetConfig()
	config.Replace(cfg)

	// Load category mapping if classifier is enabled
	var categoryMapping *classification.CategoryMapping
	if cfg.CategoryMappingPath != "" {
		if _, err := os.Stat(cfg.CategoryMappingPath); err == nil {
			categoryMapping, err = classification.LoadCategoryMapping(cfg.CategoryMappingPath)
			if err != nil {
				return nil, fmt.Errorf("failed to load category mapping: %w", err)
			}
			logging.Infof("Loaded category mapping with %d categories", categoryMapping.GetCategoryCount())
		} else {
			logging.Warnf("Category mapping file not found: %s (category classification will be disabled)", cfg.CategoryMappingPath)
		}
	}

	// Load PII mapping if PII classifier is enabled
	var piiMapping *classification.PIIMapping
	if cfg.PIIMappingPath != "" {
		if _, err := os.Stat(cfg.PIIMappingPath); err == nil {
			piiMapping, err = classification.LoadPIIMapping(cfg.PIIMappingPath)
			if err != nil {
				return nil, fmt.Errorf("failed to load PII mapping: %w", err)
			}
			logging.Infof("Loaded PII mapping with %d PII types", piiMapping.GetPIITypeCount())
		} else {
			logging.Warnf("PII mapping file not found: %s (PII classification will be disabled)", cfg.PIIMappingPath)
		}
	}

	// Load jailbreak mapping if prompt guard is enabled
	var jailbreakMapping *classification.JailbreakMapping
	if cfg.IsPromptGuardEnabled() {
		if _, err := os.Stat(cfg.PromptGuard.JailbreakMappingPath); err == nil {
			jailbreakMapping, err = classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)
			if err != nil {
				return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
			}
			logging.Infof("Loaded jailbreak mapping with %d jailbreak types", jailbreakMapping.GetJailbreakTypeCount())
		} else {
			logging.Warnf("Jailbreak mapping file not found: %s (jailbreak detection will be disabled)", cfg.PromptGuard.JailbreakMappingPath)
		}
	}

	// Initialize the BERT model for similarity search (optional)
	if cfg.BertModel.ModelID != "" {
		if initErr := candle_binding.InitModel(cfg.BertModel.ModelID, cfg.BertModel.UseCPU); initErr != nil {
			logging.Warnf("Failed to initialize BERT model: %v (semantic similarity features will be disabled)", initErr)
			// Continue without BERT model - some features will be disabled
		} else {
			logging.Infof("BERT model initialized successfully: %s", cfg.BertModel.ModelID)
		}
	} else {
		logging.Warnf("BERT model not configured (semantic similarity features will be disabled)")
	}

	categoryDescriptions := cfg.GetCategoryDescriptions()
	logging.Infof("Category descriptions: %v", categoryDescriptions)

	// Create semantic cache with config options
	cacheConfig := cache.CacheConfig{
		BackendType:         cache.CacheBackendType(cfg.SemanticCache.BackendType),
		Enabled:             cfg.SemanticCache.Enabled,
		SimilarityThreshold: cfg.GetCacheSimilarityThreshold(),
		MaxEntries:          cfg.SemanticCache.MaxEntries,
		TTLSeconds:          cfg.SemanticCache.TTLSeconds,
		EvictionPolicy:      cache.EvictionPolicyType(cfg.SemanticCache.EvictionPolicy),
		BackendConfigPath:   cfg.SemanticCache.BackendConfigPath,
		EmbeddingModel:      cfg.SemanticCache.EmbeddingModel,
	}

	// Use default backend type if not specified
	if cacheConfig.BackendType == "" {
		cacheConfig.BackendType = cache.InMemoryCacheType
	}

	semanticCache, err := cache.NewCacheBackend(cacheConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create semantic cache: %w", err)
	}

	if semanticCache.IsEnabled() {
		logging.Infof("Semantic cache enabled (backend: %s) with threshold: %.4f, TTL: %d seconds",
			cacheConfig.BackendType, cacheConfig.SimilarityThreshold, cacheConfig.TTLSeconds)
		if cacheConfig.BackendType == cache.InMemoryCacheType {
			logging.Infof("In-memory cache max entries: %d", cacheConfig.MaxEntries)
		}
	} else {
		logging.Infof("Semantic cache is disabled")
	}

	// Create tools database with config options
	toolsThreshold := cfg.BertModel.Threshold // Default to BERT threshold
	if cfg.Tools.SimilarityThreshold != nil {
		toolsThreshold = *cfg.Tools.SimilarityThreshold
	}
	toolsOptions := tools.ToolsDatabaseOptions{
		SimilarityThreshold: toolsThreshold,
		Enabled:             cfg.Tools.Enabled,
	}
	toolsDatabase := tools.NewToolsDatabase(toolsOptions)

	// Load tools from file if enabled and path is provided
	if toolsDatabase.IsEnabled() && cfg.Tools.ToolsDBPath != "" {
		if loadErr := toolsDatabase.LoadToolsFromFile(cfg.Tools.ToolsDBPath); loadErr != nil {
			logging.Warnf("Failed to load tools from file %s: %v", cfg.Tools.ToolsDBPath, loadErr)
		}
		logging.Infof("Tools database enabled with threshold: %.4f, top-k: %d",
			toolsThreshold, cfg.Tools.TopK)
	} else {
		logging.Infof("Tools database is disabled")
	}

	// Create utility components
	piiChecker := pii.NewPolicyChecker(cfg, cfg.ModelConfig)

	classifier, err := classification.NewClassifier(cfg, categoryMapping, piiMapping, jailbreakMapping)
	if err != nil {
		return nil, fmt.Errorf("failed to create classifier: %w", err)
	}

	// Create global classification service for API access with auto-discovery
	// This will prioritize LoRA models over legacy ModernBERT
	autoSvc, err := services.NewClassificationServiceWithAutoDiscovery(cfg)
	if err != nil {
		logging.Warnf("Auto-discovery failed during router initialization: %v, using legacy classifier", err)
		services.NewClassificationService(classifier, cfg)
	} else {
		logging.Infof("Router initialization: Using auto-discovered unified classifier")
		// The service is already set as global in NewUnifiedClassificationService
		_ = autoSvc
	}

	router := &OpenAIRouter{
		Config:               cfg,
		CategoryDescriptions: categoryDescriptions,
		Classifier:           classifier,
		PIIChecker:           piiChecker,
		Cache:                semanticCache,
		ToolsDatabase:        toolsDatabase,
	}

	// Log reasoning configuration after router is created
	router.logReasoningConfiguration()

	return router, nil
}
