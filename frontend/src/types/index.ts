export type UserRole = 'ADMIN' | 'REVIEWER' | 'VIEWER';

export interface User {
  id: string;
  email: string;
  full_name?: string | null;
  role: UserRole;
  is_active: boolean;
  last_login_at?: string | null;
  created_at: string;
  updated_at: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token?: string | null;
  token_type: string;
  expires_in: number;
  user: User;
}

export type JobStatus = 'QUEUED' | 'PROCESSING' | 'RETRYING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';

export interface ProcessingJob {
  id: string;
  document_id: string;
  task_id?: string | null;
  idempotency_key: string;
  extraction_method: string;
  status: JobStatus;
  attempts: number;
  max_attempts: number;
  request_review: boolean;
  started_at?: string | null;
  completed_at?: string | null;
  heartbeat_at?: string | null;
  error_code?: string | null;
  error_message?: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentStatusResponse {
  document_id: string;
  document_status: DocumentStatus;
  review_status: ReviewStatus;
  needs_review: boolean;
  error_message?: string | null;
  latest_job?: ProcessingJob | null;
  jobs: ProcessingJob[];
}

export interface ProcessingJobMetrics {
  total_jobs: number;
  queued: number;
  processing: number;
  retrying: number;
  completed: number;
  failed: number;
  cancelled: number;
  stale_running_jobs: number;
  unpublished_outbox_events: number;
  average_duration_seconds?: number | null;
  by_extraction_method: Record<string, number>;
}

export type DocumentStatus = 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export type ReviewStatus =
  | 'NOT_REQUIRED'
  | 'PENDING_REVIEW'
  | 'IN_REVIEW'
  | 'APPROVED'
  | 'REJECTED';

export type ExtractionMethod = 'PLACEHOLDER' | 'TESSERACT_BASELINE' | 'LAYOUTLMV3';

export interface LineItem {
  description?: string | null;
  quantity?: number | string | null;
  unit_price?: number | string | null;
  total_price?: number | string | null;
  experimental?: boolean;
}

export interface StructuredExtractionData {
  vendor_name?: string | null;
  vendor_address?: string | null;
  document_date?: string | null;
  total_amount?: number | string | null;
  currency?: string | null;
  tax_amount?: number | string | null;
  line_items?: LineItem[];
}

export interface ExtractionResult {
  id: string;
  document_id: string;
  extraction_method: string;
  structured_data: StructuredExtractionData | null;
  confidence: number | null;
  field_confidences?: Record<string, number> | null;
  raw_text?: string | null;
  validation_errors: string[];
  extraction_metadata?: Record<string, unknown> | null;
  created_at: string;
}

export interface DocumentItem {
  id: string;
  owner_id?: string | null;
  original_filename: string;
  content_type: string;
  file_size_bytes: number;
  status: DocumentStatus;
  review_status: ReviewStatus;
  needs_review: boolean;
  review_reasons: string[];
  error_message?: string | null;
  created_at: string;
  updated_at: string;
  extractions: ExtractionResult[];
  latest_job?: ProcessingJob | null;
}

export interface DocumentListResponse {
  items: DocumentItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface DocumentUploadResponse {
  document: DocumentItem;
  job?: ProcessingJob | null;
  message: string;
}

export interface DashboardMetrics {
  total_documents: number;
  pending: number;
  processing: number;
  completed: number;
  failed: number;
}

export interface HealthStatus {
  status: string;
  service: string;
  environment?: string;
  timestamp?: string;
  database?: string;
  redis?: string;
  worker?: string;
  tesseract_available?: boolean;
  layoutlmv3_ready?: boolean;
  layoutlmv3_is_production_ready?: boolean;
  components?: {
    database: string;
    storage: string;
  };
  phase?: number | string;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface ReviewLineItemPayload {
  description?: string | null;
  quantity?: string | number | null;
  unit_price?: string | number | null;
  total_price?: string | number | null;
}

export interface ReviewCorrectionPayload {
  vendor_name?: string | null;
  vendor_address?: string | null;
  document_date?: string | null;
  total_amount?: string | number | null;
  currency?: string | null;
  tax_amount?: string | number | null;
  line_items?: ReviewLineItemPayload[];
}

export interface DocumentReview {
  id?: string | null;
  document_id: string;
  extraction_result_id?: string | null;
  review_status: ReviewStatus;
  document_status: DocumentStatus;
  extraction_method?: string | null;
  confidence?: number | null;
  field_confidences?: Record<string, number> | null;
  validation_errors: string[];
  review_reasons: string[];
  missing_critical_fields: string[];
  original_data?: StructuredExtractionData | null;
  corrected_data?: StructuredExtractionData | null;
  changed_fields: string[];
  raw_text?: string | null;
  reviewer_identifier: string;
  reviewer_notes?: string | null;
  version: number;
  created_at?: string | null;
  updated_at?: string | null;
  approved_at?: string | null;
}

export interface ReviewQueueItem {
  document_id: string;
  review_id?: string | null;
  original_filename: string;
  content_type: string;
  file_size_bytes: number;
  document_status: DocumentStatus;
  review_status: ReviewStatus;
  needs_review: boolean;
  review_reasons: string[];
  missing_critical_fields: string[];
  extraction_method?: string | null;
  confidence?: number | null;
  validation_errors_count: number;
  changed_fields: string[];
  reviewer_identifier: string;
  version: number;
  created_at: string;
  updated_at: string;
  approved_at?: string | null;
}

export interface ReviewListResponse {
  items: ReviewQueueItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface FieldCorrectionItem {
  id: string;
  review_id: string;
  field_name: string;
  original_value?: string | null;
  corrected_value?: string | null;
  created_at: string;
}

export interface ReviewAuditEventItem {
  id: string;
  review_id: string;
  action: string;
  from_status?: string | null;
  to_status?: string | null;
  changed_fields: string[];
  actor_identifier: string;
  notes?: string | null;
  event_timestamp: string;
}

export interface ReviewHistoryResponse {
  document_id: string;
  review_id?: string | null;
  review_status: ReviewStatus;
  field_corrections: FieldCorrectionItem[];
  audit_events: ReviewAuditEventItem[];
}

export interface ReviewMetrics {
  total_documents: number;
  not_required: number;
  pending_review: number;
  in_review: number;
  awaiting_review: number;
  approved: number;
  rejected: number;
  avg_corrected_fields: number;
  review_completion_rate: number;
  eligible_for_feedback_export: number;
  review_confidence_threshold: number;
}

export interface PreviewMeta {
  document_id: string;
  original_filename: string;
  content_type: string;
  is_pdf: boolean;
  total_pages: number;
  max_preview_pages: number;
  preview_url: string;
}

export interface ExtractionMethodReadiness {
  method: string;
  label: string;
  available: boolean;
  is_default: boolean;
  is_production_ready: boolean;
  badge: string;
  description: string;
  unavailable_reason?: string | null;
  checkpoint_identity?: string | null;
  metadata?: Record<string, unknown> | null;
}

export interface ModelReadinessResponse {
  default_method: string;
  ml_inference_enabled: boolean;
  torch_runtime_available: boolean;
  tesseract_available: boolean;
  tesseract_version?: string | null;
  tesseract_message?: string | null;
  layoutlmv3_ready: boolean;
  layoutlmv3_is_production_ready: boolean;
  layoutlmv3_reason?: string | null;
  checkpoint_identity?: string | null;
  checkpoint_size_mb?: number | null;
  checkpoint_metadata?: Record<string, unknown> | null;
  methods: ExtractionMethodReadiness[];
}
