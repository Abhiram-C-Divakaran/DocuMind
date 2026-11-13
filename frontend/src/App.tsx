import React, { useState, useEffect, useCallback, useRef } from 'react';
import { apiClient } from './api/client';
import { authStorage } from './api/authStorage';
import {
  DocumentItem,
  DashboardMetrics,
  HealthStatus,
  DocumentUploadResponse,
  ReviewMetrics,
  User,
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
import { LoginPanel } from './components/Auth/LoginPanel';
import { UserManagementView } from './components/Auth/UserManagementView';
import { Toast, ToastMessage } from './components/Common/Toast';
import { UploadCloud, FileText, Loader2 } from 'lucide-react';

export const App: React.FC = () => {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [checkingAuth, setCheckingAuth] = useState<boolean>(true);
  const [sessionNotice, setSessionNotice] = useState<string | null>(null);

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

  const [page, setPage] = useState<number>(1);
  const [pageSize] = useState<number>(10);
  const [search, setSearch] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('');

  const [selectedDoc, setSelectedDoc] = useState<DocumentItem | null>(null);
  const [toast, setToast] = useState<ToastMessage | null>(null);

  const pollDelayRef = useRef<number>(2000);

  const showToast = useCallback((type: 'success' | 'error' | 'info', title: string, message?: string) => {
    setToast({
      id: Date.now().toString(),
      type,
      title,
      message,
    });
  }, []);

  useEffect(() => {
    apiClient.setSessionExpiredHandler(() => {
      setCurrentUser(null);
      setSessionNotice('Your session has expired. Please sign in again.');
    });
    return () => {
      apiClient.setSessionExpiredHandler(null);
    };
  }, []);

  useEffect(() => {
    let active = true;
    apiClient
      .restoreSession()
      .then((user) => {
        if (active) setCurrentUser(user);
      })
      .catch(() => {
        authStorage.clear();
        if (active) setCurrentUser(null);
      })
      .finally(() => {
        if (active) setCheckingAuth(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const loadHealth = useCallback(async () => {
    try {
      const data = await apiClient.checkHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    }
  }, []);

  const loadMetrics = useCallback(async () => {
    if (!currentUser) return;
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
  }, [currentUser]);

  const loadDocuments = useCallback(async () => {
    if (!currentUser) return;
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
  }, [currentUser, page, pageSize, statusFilter, search, showToast]);

  const handleRefreshAll = async () => {
    setIsRefreshing(true);
    await Promise.all([loadHealth(), loadMetrics(), loadDocuments()]);
    setReviewRefreshSignal((n) => n + 1);
    setIsRefreshing(false);
    showToast('info', 'Data Refreshed', 'Latest documents and metrics synchronized.');
  };

  useEffect(() => {
    loadHealth();
    if (currentUser) {
      loadMetrics();
      loadDocuments();
    }
  }, [currentUser, loadHealth, loadMetrics, loadDocuments]);

  // Bounded exponential backoff polling while documents are queued/processing; cleans up on unmount
  useEffect(() => {
    if (!currentUser) return;
    const hasActive = documents.some(
      (d) => d.status === 'PENDING' || d.status === 'PROCESSING'
    );
    if (!hasActive) {
      pollDelayRef.current = 2000;
      return;
    }

    let cancelled = false;
    const delay = pollDelayRef.current;
    const timer = window.setTimeout(async () => {
      if (cancelled) return;
      await Promise.all([loadDocuments(), loadMetrics()]);
      pollDelayRef.current = Math.min(10000, Math.round(delay * 1.5));
    }, delay);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [currentUser, documents, loadDocuments, loadMetrics]);

  const handleLogout = async () => {
    await apiClient.logout();
    setCurrentUser(null);
    setSelectedDoc(null);
    setActiveReviewDocId(null);
    setCurrentTab('dashboard');
    setSessionNotice(null);
  };

  const handleUploadSuccess = (response: DocumentUploadResponse) => {
    pollDelayRef.current = 1500;
    showToast('success', 'Document Queued (202)', response.message);
    loadMetrics();
    loadDocuments();
    setReviewRefreshSignal((n) => n + 1);
    setCurrentTab('documents');
  };

  const handleRetryDocument = async (doc: DocumentItem) => {
    try {
      const res = await apiClient.retryDocument(doc.id);
      pollDelayRef.current = 1500;
      showToast('info', 'Job Re-queued', res.message);
      await Promise.all([loadDocuments(), loadMetrics()]);
      if (selectedDoc?.id === doc.id) {
        setSelectedDoc(res.document);
      }
    } catch (err: unknown) {
      showToast('error', 'Retry Failed', (err as Error).message);
    }
  };

  const handleCancelDocument = async (doc: DocumentItem) => {
    try {
      await apiClient.cancelDocument(doc.id);
      showToast('info', 'Job Cancelled', `Stopped queued processing for '${doc.original_filename}'.`);
      await Promise.all([loadDocuments(), loadMetrics()]);
    } catch (err: unknown) {
      showToast('error', 'Cancel Failed', (err as Error).message);
    }
  };

  const handleDeleteDocument = async (doc: DocumentItem) => {
    try {
      await apiClient.deleteDocument(doc.id);
      showToast('success', 'Document Deleted', `Removed '${doc.original_filename}'.`);
      if (selectedDoc?.id === doc.id) setSelectedDoc(null);
      await Promise.all([loadDocuments(), loadMetrics()]);
    } catch (err: unknown) {
      showToast('error', 'Delete Failed', (err as Error).message);
    }
  };

  const handleOpenReviewWorkspace = (documentId: string) => {
    setActiveReviewDocId(documentId);
    setCurrentTab('reviews');
  };

  if (checkingAuth) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center text-slate-300 text-sm gap-2.5">
        <Loader2 className="w-5 h-5 animate-spin text-indigo-400" />
        <span>Verifying session...</span>
      </div>
    );
  }

  if (!currentUser) {
    return (
      <LoginPanel
        sessionNotice={sessionNotice}
        onAuthenticated={(user) => {
          setSessionNotice(null);
          setCurrentUser(user);
          setCurrentTab('dashboard');
        }}
      />
    );
  }

  const canMutate = currentUser.role === 'ADMIN' || currentUser.role === 'REVIEWER';

  return (
    <div className="flex h-screen bg-slate-100 overflow-hidden text-slate-800">
      <Sidebar
        currentTab={currentTab}
        userRole={currentUser.role}
        onTabChange={(tab) => {
          setCurrentTab(tab);
          if (tab !== 'reviews') {
            setActiveReviewDocId(null);
          }
        }}
        health={health}
        awaitingReviewCount={reviewMetrics?.awaiting_review ?? 0}
      />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Header
          currentTab={currentTab}
          currentUser={currentUser}
          onLogout={handleLogout}
          onOpenUpload={() => setCurrentTab('upload')}
          onRefresh={handleRefreshAll}
          isRefreshing={isRefreshing}
        />

        <main className="flex-1 overflow-y-auto p-6 lg:p-8 space-y-6">
          {currentTab === 'dashboard' && (
            <div className="space-y-6 max-w-7xl mx-auto">
              <MetricsOverview
                metrics={metrics}
                reviewMetrics={reviewMetrics}
                loading={loadingMetrics}
                onOpenReviewQueue={() => {
                  setActiveReviewDocId(null);
                  setCurrentTab('reviews');
                }}
              />

              <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900">Recent Ingestion Activity</h3>
                    <p className="text-xs text-slate-500">Latest documents processed asynchronously</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => setCurrentTab('documents')}
                      className="text-xs font-semibold text-indigo-600 hover:text-indigo-700 hover:underline"
                    >
                      View Full Ledger →
                    </button>
                  </div>
                </div>

                {documents.length > 0 ? (
                  <DocumentTable
                    documents={documents.slice(0, 5)}
                    total={Math.min(documents.length, 5)}
                    page={1}
                    pageSize={5}
                    loading={loadingDocs}
                    search=""
                    statusFilter=""
                    onSearchChange={() => {}}
                    onStatusFilterChange={() => {}}
                    onPageChange={() => {}}
                    onSelectDocument={(doc) => setSelectedDoc(doc)}
                    onReviewDocument={canMutate ? (doc) => handleOpenReviewWorkspace(doc.id) : undefined}
                    onRetryDocument={canMutate ? handleRetryDocument : undefined}
                    onCancelDocument={canMutate ? handleCancelDocument : undefined}
                    onDeleteDocument={canMutate ? handleDeleteDocument : undefined}
                    canMutate={canMutate}
                    onUploadClick={() => canMutate && setCurrentTab('upload')}
                  />
                ) : (
                  <div className="p-8 text-center text-slate-400 text-xs">
                    <FileText className="w-8 h-8 mx-auto mb-2 text-slate-300" />
                    <p className="font-medium text-slate-600 mb-1">No documents ingested yet</p>
                    {canMutate && (
                      <button
                        onClick={() => setCurrentTab('upload')}
                        className="mt-2 inline-flex items-center gap-1.5 px-3 py-1.5 bg-indigo-50 text-indigo-700 font-semibold rounded-md hover:bg-indigo-100 transition-colors"
                      >
                        <UploadCloud className="w-3.5 h-3.5" /> Upload First Receipt
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {currentTab === 'documents' && (
            <div className="max-w-7xl mx-auto">
              <DocumentTable
                documents={documents}
                total={totalDocs}
                page={page}
                pageSize={pageSize}
                loading={loadingDocs}
                search={search}
                statusFilter={statusFilter}
                onSearchChange={(q) => {
                  setSearch(q);
                  setPage(1);
                }}
                onStatusFilterChange={(s) => {
                  setStatusFilter(s);
                  setPage(1);
                }}
                onPageChange={(p) => setPage(p)}
                onSelectDocument={(doc) => setSelectedDoc(doc)}
                onReviewDocument={canMutate ? (doc) => handleOpenReviewWorkspace(doc.id) : undefined}
                onRetryDocument={canMutate ? handleRetryDocument : undefined}
                onCancelDocument={canMutate ? handleCancelDocument : undefined}
                onDeleteDocument={canMutate ? handleDeleteDocument : undefined}
                canMutate={canMutate}
                onUploadClick={() => canMutate && setCurrentTab('upload')}
              />
            </div>
          )}

          {currentTab === 'reviews' && (
            <div className="max-w-7xl mx-auto">
              {activeReviewDocId ? (
                <DocumentReviewWorkspace
                  documentId={activeReviewDocId}
                  readOnly={!canMutate}
                  onBack={() => {
                    setActiveReviewDocId(null);
                    loadMetrics();
                    loadDocuments();
                  }}
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

          {currentTab === 'upload' && canMutate && (
            <div className="max-w-4xl mx-auto py-4">
              <DocumentUploader
                onUploadSuccess={handleUploadSuccess}
                onJobFinished={() => {
                  loadMetrics();
                  loadDocuments();
                  setReviewRefreshSignal((n) => n + 1);
                }}
                onCancel={() => setCurrentTab('dashboard')}
              />
            </div>
          )}

          {currentTab === 'ml-architecture' && (
            <div className="max-w-7xl mx-auto">
              <MlArchitectureView />
            </div>
          )}

          {currentTab === 'users' && currentUser.role === 'ADMIN' && (
            <div className="max-w-7xl mx-auto">
              <UserManagementView
                currentUser={currentUser}
                health={health}
                onRefreshHealth={loadHealth}
                onToast={(type, msg) => showToast(type, msg)}
              />
            </div>
          )}
        </main>
      </div>

      <DocumentDetailModal
        document={selectedDoc}
        onClose={() => setSelectedDoc(null)}
        canMutate={canMutate}
        onRetryDocument={canMutate ? handleRetryDocument : undefined}
        onOpenReview={
          canMutate
            ? (docId) => {
                setSelectedDoc(null);
                handleOpenReviewWorkspace(docId);
              }
            : undefined
        }
      />

      <Toast toast={toast} onClose={() => setToast(null)} />
    </div>
  );
};

export default App;
