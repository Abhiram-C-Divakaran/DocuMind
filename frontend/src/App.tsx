import React, { useState, useEffect, useCallback } from 'react';
import { apiClient } from './api/client';
import {
  DocumentItem,
  DashboardMetrics,
  HealthStatus,
  DocumentUploadResponse,
  ReviewMetrics,
} from './types';
import { Sidebar, NavTab } from './components/Layout/Sidebar';
import { Header } from './components/Layout/Header';
import { MetricsOverview } from './components/Dashboard/MetricsOverview';
import { DocumentTable } from './components/Documents/DocumentTable';
import { DocumentUploader } from './components/Documents/DocumentUploader';
import { DocumentDetailModal } from './components/Documents/DocumentDetailModal';
import { MlArchitectureView } from './components/Dashboard/MlArchitectureView';
import { ReviewQueue } from './components/Reviews/ReviewQueue';
import { DocumentReviewWorkspace } from './components/Reviews/DocumentReviewWorkspace';
import { Toast, ToastMessage } from './components/Common/Toast';
import { UploadCloud, FileText } from 'lucide-react';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>('dashboard');
  const [activeReviewDocId, setActiveReviewDocId] = useState<string | null>(null);
  const [reviewRefreshSignal, setReviewRefreshSignal] = useState<number>(0);

  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [totalDocs, setTotalDocs] = useState<number>(0);
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [reviewMetrics, setReviewMetrics] = useState<ReviewMetrics | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);

  const [loadingDocs, setLoadingDocs] = useState<boolean>(true);
  const [loadingMetrics, setLoadingMetrics] = useState<boolean>(true);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  // Filters & Pagination
  const [page, setPage] = useState<number>(1);
  const [pageSize] = useState<number>(10);
  const [search, setSearch] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('');

  // Modals & Notifications
  const [selectedDoc, setSelectedDoc] = useState<DocumentItem | null>(null);
  const [toast, setToast] = useState<ToastMessage | null>(null);

  const showToast = (type: 'success' | 'error' | 'info', title: string, message?: string) => {
    setToast({
      id: Date.now().toString(),
      type,
      title,
      message,
    });
  };

  // Fetch Health
  const loadHealth = useCallback(async () => {
    try {
      const data = await apiClient.checkHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    }
  }, []);

  // Fetch Metrics
  const loadMetrics = useCallback(async () => {
    setLoadingMetrics(true);
    try {
      const [docStats, revStats] = await Promise.all([
        apiClient.fetchMetrics(),
        apiClient.fetchReviewMetrics().catch(() => null),
      ]);
      setMetrics(docStats);
      if (revStats) {
        setReviewMetrics(revStats);
      }
    } catch (err: unknown) {
      console.error('Failed to load metrics:', err);
    } finally {
      setLoadingMetrics(false);
    }
  }, []);

  // Fetch Documents
  const loadDocuments = useCallback(async () => {
    setLoadingDocs(true);
    try {
      const data = await apiClient.fetchDocuments({
        page,
        page_size: pageSize,
        status: statusFilter || undefined,
        search: search.trim() || undefined,
      });
      setDocuments(data.items);
      setTotalDocs(data.total);
    } catch (err: unknown) {
      console.error('Failed to load documents:', err);
      showToast('error', 'Failed to fetch documents', (err as Error).message);
    } finally {
      setLoadingDocs(false);
    }
  }, [page, pageSize, statusFilter, search]);

  const handleRefreshAll = async () => {
    setIsRefreshing(true);
    await Promise.all([loadHealth(), loadMetrics(), loadDocuments()]);
    setReviewRefreshSignal((n) => n + 1);
    setIsRefreshing(false);
    showToast('info', 'Data Refreshed', 'Latest documents and metrics synchronized.');
  };

  useEffect(() => {
    loadHealth();
    loadMetrics();
  }, [loadHealth, loadMetrics]);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  const handleUploadSuccess = (response: DocumentUploadResponse) => {
    showToast('success', 'Document Ingested', response.message);
    loadMetrics();
    loadDocuments();
    setReviewRefreshSignal((n) => n + 1);
    setCurrentTab('documents');
  };

  const handleOpenReviewWorkspace = (documentId: string) => {
    setActiveReviewDocId(documentId);
    setCurrentTab('reviews');
  };

  return (
    <div className="flex h-screen bg-slate-100 overflow-hidden text-slate-800">
      {/* Sidebar */}
      <Sidebar
        currentTab={currentTab}
        onTabChange={(tab) => {
          if (tab !== 'reviews') {
            setActiveReviewDocId(null);
          }
          setCurrentTab(tab);
        }}
        health={health}
        awaitingReviewCount={reviewMetrics?.awaiting_review ?? 0}
      />

      {/* Main Container */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <Header
          currentTab={currentTab}
          onOpenUpload={() => setCurrentTab('upload')}
          onRefresh={handleRefreshAll}
          isRefreshing={isRefreshing}
        />

        {/* Dynamic Viewport */}
        <main className="flex-1 overflow-y-auto p-8 space-y-8">
          {currentTab === 'dashboard' && (
            <div className="space-y-8 max-w-7xl mx-auto">
              {/* Telemetry Metrics Cards */}
              <MetricsOverview
                metrics={metrics}
                reviewMetrics={reviewMetrics}
                loading={loadingMetrics}
                onOpenReviewQueue={() => {
                  setActiveReviewDocId(null);
                  setCurrentTab('reviews');
                }}
              />

              {/* Quick Actions & Recent Docs Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Recent Documents Table */}
                <div className="lg:col-span-2 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2">
                      <FileText className="w-4 h-4 text-indigo-600" />
                      <span>Recent Documents Ledger</span>
                    </h3>
                    <button
                      onClick={() => setCurrentTab('documents')}
                      className="text-xs font-semibold text-indigo-600 hover:text-indigo-800"
                    >
                      View all &rarr;
                    </button>
                  </div>

                  <DocumentTable
                    documents={documents.slice(0, 5)}
                    total={totalDocs}
                    page={page}
                    pageSize={pageSize}
                    loading={loadingDocs}
                    search={search}
                    statusFilter={statusFilter}
                    onSearchChange={setSearch}
                    onStatusFilterChange={setStatusFilter}
                    onPageChange={setPage}
                    onSelectDocument={(doc) => setSelectedDoc(doc)}
                    onReviewDocument={(doc) => handleOpenReviewWorkspace(doc.id)}
                    onUploadClick={() => setCurrentTab('upload')}
                  />
                </div>

                {/* Upload Quick Launcher */}
                <div className="space-y-3">
                  <h3 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2">
                    <UploadCloud className="w-4 h-4 text-indigo-600" />
                    <span>Quick Ingest</span>
                  </h3>
                  <DocumentUploader onUploadSuccess={handleUploadSuccess} />
                </div>
              </div>
            </div>
          )}

          {currentTab === 'documents' && (
            <div className="space-y-6 max-w-7xl mx-auto">
              <DocumentTable
                documents={documents}
                total={totalDocs}
                page={page}
                pageSize={pageSize}
                loading={loadingDocs}
                search={search}
                statusFilter={statusFilter}
                onSearchChange={setSearch}
                onStatusFilterChange={setStatusFilter}
                onPageChange={setPage}
                onSelectDocument={(doc) => setSelectedDoc(doc)}
                onReviewDocument={(doc) => handleOpenReviewWorkspace(doc.id)}
                onUploadClick={() => setCurrentTab('upload')}
              />
            </div>
          )}

          {currentTab === 'reviews' && (
            <div className="space-y-6 max-w-7xl mx-auto">
              {activeReviewDocId ? (
                <DocumentReviewWorkspace
                  documentId={activeReviewDocId}
                  onBack={() => setActiveReviewDocId(null)}
                  onReviewUpdated={() => {
                    loadMetrics();
                    loadDocuments();
                    setReviewRefreshSignal((n) => n + 1);
                  }}
                />
              ) : (
                <ReviewQueue
                  onOpenReview={(docId) => setActiveReviewDocId(docId)}
                  refreshSignal={reviewRefreshSignal}
                />
              )}
            </div>
          )}

          {currentTab === 'upload' && (
            <div className="py-6">
              <DocumentUploader
                onUploadSuccess={handleUploadSuccess}
                onCancel={() => setCurrentTab('documents')}
              />
            </div>
          )}

          {currentTab === 'ml-architecture' && <MlArchitectureView />}
        </main>
      </div>

      {/* Document Inspector Modal */}
      <DocumentDetailModal
        document={selectedDoc}
        onClose={() => setSelectedDoc(null)}
        onOpenReview={handleOpenReviewWorkspace}
      />

      {/* Global Toast */}
      <Toast toast={toast} onClose={() => setToast(null)} />
    </div>
  );
};

