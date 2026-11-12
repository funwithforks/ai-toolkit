'use client';

import { useMemo, useState } from 'react';
import { Modal } from '@/components/Modal';
import Link from 'next/link';
import { TextInput } from '@/components/formInputs';
import useDatasetList, { DatasetCounts } from '@/hooks/useDatasetList';
import { Button } from '@headlessui/react';
import { FaFilm, FaRegImage, FaRegTrashAlt, FaVolumeUp } from 'react-icons/fa';
import { openConfirm } from '@/components/ConfirmModal';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { apiClient } from '@/utils/api';
import { useRouter } from 'next/navigation';
import classNames from 'classnames';

type SortOrder = 'name-asc' | 'name-desc' | 'items-asc' | 'items-desc';

const SORT_OPTIONS: { value: SortOrder; label: string }[] = [
  { value: 'name-asc', label: 'Name ↑' },
  { value: 'name-desc', label: 'Name ↓' },
  { value: 'items-desc', label: 'Items ↓' },
  { value: 'items-asc', label: 'Items ↑' },
];

function TypeToggle({
  icon: Icon,
  label,
  checked,
  onChange,
}: {
  icon: React.ComponentType;
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <button
      type="button"
      title={`${checked ? 'Hide' : 'Show'} ${label}`}
      onClick={() => onChange(!checked)}
      className={classNames(
        'p-1.5 rounded-sm border transition-colors',
        checked ? 'border-gray-600 bg-gray-800 text-gray-200' : 'border-gray-800 bg-transparent text-gray-600',
      )}
    >
      <Icon />
    </button>
  );
}

function ItemCountBadge({ counts }: { counts: DatasetCounts | null }) {
  if (!counts) {
    return <span className="text-xs text-gray-500">—</span>;
  }
  return (
    <span className="inline-flex items-center gap-2">
      <span className="bg-green-900/60 text-green-400 border border-green-700/60 rounded-full px-2 py-0.5 text-xs font-medium">
        {counts.total}
      </span>
      {counts.image > 0 && (
        <span className="inline-flex items-center gap-1 text-xs text-gray-400">
          <FaRegImage /> {counts.image}
        </span>
      )}
      {counts.video > 0 && (
        <span className="inline-flex items-center gap-1 text-xs text-gray-400">
          <FaFilm /> {counts.video}
        </span>
      )}
      {counts.audio > 0 && (
        <span className="inline-flex items-center gap-1 text-xs text-gray-400">
          <FaVolumeUp /> {counts.audio}
        </span>
      )}
    </span>
  );
}

export default function Datasets() {
  const router = useRouter();
  const { datasets, status, refreshDatasets } = useDatasetList();
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isNewDatasetModalOpen, setIsNewDatasetModalOpen] = useState(false);
  const [filterText, setFilterText] = useState('');
  const [visibleTypes, setVisibleTypes] = useState({ image: true, video: true, audio: true });
  const [sortOrder, setSortOrder] = useState<SortOrder>('name-asc');

  // Transform datasets array into rows with objects, applying the name
  // filter, the type visibility toggles, and the sort order.
  const tableRows = useMemo(() => {
    const filter = filterText.trim().toLowerCase();
    const anyTypeChecked = visibleTypes.image || visibleTypes.video || visibleTypes.audio;
    const rows = datasets
      .filter(dataset => !filter || dataset.name.toLowerCase().includes(filter))
      .filter(
        dataset =>
          !anyTypeChecked ||
          (dataset.counts !== null &&
            ((visibleTypes.image && dataset.counts.image > 0) ||
              (visibleTypes.video && dataset.counts.video > 0) ||
              (visibleTypes.audio && dataset.counts.audio > 0))),
      );
    rows.sort((a, b) => {
      switch (sortOrder) {
        case 'name-asc':
          return a.name.localeCompare(b.name);
        case 'name-desc':
          return b.name.localeCompare(a.name);
        case 'items-asc':
          return (a.counts?.total ?? -1) - (b.counts?.total ?? -1) || a.name.localeCompare(b.name);
        case 'items-desc':
          return (b.counts?.total ?? -1) - (a.counts?.total ?? -1) || a.name.localeCompare(b.name);
      }
    });
    return rows.map(dataset => ({ name: dataset.name, counts: dataset.counts }));
  }, [datasets, filterText, visibleTypes, sortOrder]);

  const columns: TableColumn[] = [
    {
      title: 'Dataset Name',
      key: 'name',
      render: row => (
        <Link href={`/datasets/${row.name}`} className="text-gray-200 hover:text-gray-100">
          {row.name}
        </Link>
      ),
    },
    {
      title: 'Items',
      key: 'counts',
      render: row => <ItemCountBadge counts={row.counts} />,
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'w-20 text-right',
      render: row => (
        <button
          className="text-gray-200 hover:bg-red-600 p-2 rounded-full transition-colors"
          onClick={() => handleDeleteDataset(row.name)}
        >
          <FaRegTrashAlt />
        </button>
      ),
    },
  ];

  const handleDeleteDataset = (datasetName: string) => {
    openConfirm({
      title: 'Delete Dataset',
      message: `Are you sure you want to delete the dataset "${datasetName}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/datasets/delete', { name: datasetName })
          .then(() => {
            console.log('Dataset deleted:', datasetName);
            refreshDatasets();
          })
          .catch(error => {
            console.error('Error deleting dataset:', error);
          });
      },
    });
  };

  const handleCreateDataset = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const data = await apiClient.post('/api/datasets/create', { name: newDatasetName }).then(res => res.data);
      console.log('New dataset created:', data);
      refreshDatasets();
      setNewDatasetName('');
      setIsNewDatasetModalOpen(false);
    } catch (error) {
      console.error('Error creating new dataset:', error);
    }
  };

  const openNewDatasetModal = () => {
    openConfirm({
      title: 'New Dataset',
      message: 'Enter the name of the new dataset:',
      type: 'info',
      confirmText: 'Create',
      inputTitle: 'Dataset Name',
      onConfirm: async (name?: string) => {
        if (!name) {
          console.error('Dataset name is required.');
          return;
        }
        try {
          const data = await apiClient.post('/api/datasets/create', { name }).then(res => res.data);
          console.log('New dataset created:', data);
          if (data.name) {
            router.push(`/datasets/${data.name}`);
          } else {
            refreshDatasets();
          }
        } catch (error) {
          console.error('Error creating new dataset:', error);
        }
      },
    });
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">Datasets</h1>
        </div>
        <div className="flex items-center gap-2 ml-2 min-w-0">
          <input
            type="text"
            value={filterText}
            onChange={e => setFilterText(e.target.value)}
            placeholder="Filter..."
            className="w-32 sm:w-48 text-sm px-2 py-1 bg-gray-950 dark:bg-gray-800 border border-gray-700 rounded-sm text-gray-100 placeholder:text-gray-500 focus:ring-2 focus:ring-gray-600 focus:border-transparent focus:outline-none"
          />
          <div className="flex items-center gap-1">
            <TypeToggle
              icon={FaRegImage}
              label="image datasets"
              checked={visibleTypes.image}
              onChange={checked => setVisibleTypes(prev => ({ ...prev, image: checked }))}
            />
            <TypeToggle
              icon={FaFilm}
              label="video datasets"
              checked={visibleTypes.video}
              onChange={checked => setVisibleTypes(prev => ({ ...prev, video: checked }))}
            />
            <TypeToggle
              icon={FaVolumeUp}
              label="audio datasets"
              checked={visibleTypes.audio}
              onChange={checked => setVisibleTypes(prev => ({ ...prev, audio: checked }))}
            />
          </div>
          <select
            value={sortOrder}
            onChange={e => setSortOrder(e.target.value as SortOrder)}
            className="text-sm px-2 py-1 bg-gray-950 dark:bg-gray-800 border border-gray-700 rounded-sm text-gray-100 focus:outline-none"
          >
            {SORT_OPTIONS.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div className="flex-1"></div>
        <div>
          <Button
            className="text-white bg-slate-600 px-2 sm:px-3 py-1 rounded-md hover:bg-slate-500 transition-colors text-sm sm:text-base whitespace-nowrap"
            onClick={() => openNewDatasetModal()}
          >
            <span className="sm:hidden">+ New</span>
            <span className="hidden sm:inline">New Dataset</span>
          </Button>
        </div>
      </TopBar>

      <MainContent>
        <UniversalTable
          columns={columns}
          rows={tableRows}
          isLoading={status === 'loading'}
          onRefresh={refreshDatasets}
        />
      </MainContent>

      <Modal
        isOpen={isNewDatasetModalOpen}
        onClose={() => setIsNewDatasetModalOpen(false)}
        title="New Dataset"
        size="md"
      >
        <div className="space-y-4 text-gray-200">
          <form onSubmit={handleCreateDataset}>
            <div className="text-sm text-gray-400">
              This will create a new folder with the name below in your dataset folder.
            </div>
            <div className="mt-4">
              <TextInput label="Dataset Name" value={newDatasetName} onChange={value => setNewDatasetName(value)} />
            </div>

            <div className="mt-6 flex justify-end space-x-3">
              <button
                type="button"
                className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
                onClick={() => setIsNewDatasetModalOpen(false)}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                Confirm
              </button>
            </div>
          </form>
        </div>
      </Modal>
    </>
  );
}
