import Loading from './Loading';
import classNames from 'classnames';
import { memo, useEffect } from 'react';
import { perfCount } from '@/utils/perfLog';

export interface TableColumn {
  title: React.ReactNode;
  key: string;
  render?: (row: any) => React.ReactNode;
  className?: string;
}

interface TableRow {
  [key: string]: any;
}

/**
 * Memoized row: skips re-rendering when the row object or the column set is
 * unchanged. Relies on useJobsList reusing job objects across no-op polls so
 * unchanged rows keep a stable identity. Striping is done in CSS (nth-child)
 * rather than by row index so a sort-order change can't re-render every row.
 */
const MemoizedTableRow = memo(function MemoizedTableRow({
  row,
  columns,
}: {
  row: TableRow;
  columns: TableColumn[];
}) {
  useEffect(() => {
    perfCount('TableRow.render');
  });
  return (
    <tr className="odd:bg-gray-900 even:bg-gray-800 border-b border-gray-700 hover:bg-gray-700">
      {columns.map(column => (
        <td key={column.key} className={classNames('px-3 py-2', column.className)}>
          {column.render ? column.render(row) : row[column.key]}
        </td>
      ))}
    </tr>
  );
});

interface TableProps {
  columns: TableColumn[];
  rows: TableRow[];
  isLoading: boolean;
  theadClassName?: string;
  onRefresh: () => void;
}

export default function UniversalTable({
  columns,
  rows,
  isLoading,
  theadClassName = 'text-gray-400',
  onRefresh = () => {},
}: TableProps) {
  // Perf diagnostics: count committed renders of the table. Actual row
  // renders are counted inside MemoizedTableRow.
  useEffect(() => {
    perfCount('UniversalTable.render');
  });
  return (
    <div className="w-full bg-gray-900 rounded-md shadow-md">
      {isLoading ? (
        <div className="p-4 flex justify-center">
          <Loading />
        </div>
      ) : rows.length === 0 ? (
        <div className="p-6 text-center text-gray-400">
          <p className="text-sm">Empty</p>
          <button
            onClick={() => onRefresh()}
            className="mt-2 px-3 py-1 text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 rounded transition-colors"
          >
            Refresh
          </button>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left text-gray-300">
            <thead className={classNames('text-xs uppercase bg-gray-800', theadClassName)}>
              <tr>
                {columns.map(column => (
                  <th key={column.key} className="px-3 py-2">
                    {column.title}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows?.map((row, index) => (
                <MemoizedTableRow key={row.id ?? index} row={row} columns={columns} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
