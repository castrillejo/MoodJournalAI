import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, User, Loader2 } from "lucide-react";

import { searchUsers, getUserStats } from "../services/api";
import EmotionChart from "./EmotionChart";
import { EMOTION_EMOJIS } from "../constants/emotions";

const DEBOUNCE_MS = 300;

const UsersSection = () => {
    const formatEmotion = (e) => (e ? e.charAt(0).toUpperCase() + e.slice(1) : "");
    const [query, setQuery] = useState("");
    const [isSearching, setIsSearching] = useState(false);
    const [searchResults, setSearchResults] = useState([]);
    const [searchOpen, setSearchOpen] = useState(false);
    const [searchError, setSearchError] = useState(null);

    const [selectedUser, setSelectedUser] = useState(null);
    const [userStats, setUserStats] = useState(null);
    const [isLoadingStats, setIsLoadingStats] = useState(false);
    const [statsError, setStatsError] = useState(null);

    const debounceRef = useRef(null);
    const lastQueryRef = useRef("");

    const canSearch = query.trim().length >= 1;

    // --- Buscar con debounce ---
    useEffect(() => {
        setSearchError(null);

        if (!canSearch) {
            setSearchResults([]);
            setSearchOpen(false);
            return;
        }

        // debounce
        if (debounceRef.current) clearTimeout(debounceRef.current);

        debounceRef.current = setTimeout(async () => {
            const q = query.trim();
            lastQueryRef.current = q;

            setIsSearching(true);
            try {
                const res = await searchUsers(q, 5);
                if (lastQueryRef.current !== q) return;

                const results = res?.results ?? [];
                setSearchResults(results);
                setSearchOpen(true);
            } catch (e) {
                setSearchError("No se pudo buscar usuarios. Revisa que el backend esté levantado.");
                setSearchResults([]);
                setSearchOpen(false);
            } finally {
                if (lastQueryRef.current === q) setIsSearching(false);
            }
        }, DEBOUNCE_MS);

        return () => {
            if (debounceRef.current) clearTimeout(debounceRef.current);
        };
    }, [query, canSearch]);

    const handleSelectUser = async (u) => {
        setSelectedUser(u);
        setUserStats(null);
        setStatsError(null);
        setSearchOpen(false);
        setSearchResults([]);
        setQuery(u?.nombre ?? "");

        if (!u?.id_usuario) return;

        setIsLoadingStats(true);
        try {
            const data = await getUserStats(u.id_usuario);
            setUserStats(data);
        } catch (e) {
            setStatsError("No se pudieron cargar las estadísticas del usuario.");
        } finally {
            setIsLoadingStats(false);
        }
    };

    const emotionScoresForChart = useMemo(() => {
        // Preferimos charts.emotion_share porque ya viene listo para gráfica
        const arr = userStats?.charts?.emotion_share;
        if (Array.isArray(arr) && arr.length) {
            return arr
                .map(({ emotion, value }) => ({ emotion, score: Number(value) || 0 }))
                .filter((x) => x.score > 0);
        }

        // Fallback por si algún día no viene charts
        const shares = userStats?.stats?.emotion_share;
        if (!shares) return [];
        return Object.entries(shares)
            .map(([emotion, value]) => ({ emotion, score: Number(value) || 0 }))
            .filter((x) => x.score > 0);
    }, [userStats]);

    const topEmotion = userStats?.stats?.top_emotion || "";
    const topShare = userStats?.stats?.top_emotion_share || 0;

    return (
        <section className="mt-14">
            <div className="max-w-6xl mx-auto">
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                    <div className="mb-6">
                        <h2 className="text-3xl font-bold text-gray-800">👤 Usuarios</h2>
                        <p className="text-gray-600 mt-1">
                            Busca un usuario y visualiza sus estadísticas.
                        </p>
                    </div>

                    {/* Search card */}
                    <div className="bg-white rounded-2xl shadow-lg p-6 border-2 border-gray-100">
                        <div className="relative">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-xl bg-gray-100">
                                    <Search className="w-5 h-5 text-gray-700" />
                                </div>

                                <div className="flex-1">
                                    <input
                                        value={query}
                                        onChange={(e) => {
                                            setQuery(e.target.value);
                                            setSelectedUser(null);
                                            setUserStats(null);
                                            setStatsError(null);
                                        }}
                                        onFocus={() => {
                                            if (searchResults.length > 0) setSearchOpen(true);
                                        }}
                                        placeholder='Buscar por nombre'
                                        className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                    />
                                </div>

                                {isSearching && (
                                    <div className="flex items-center gap-2 text-sm text-gray-500">
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Buscando...
                                    </div>
                                )}
                            </div>

                            {/* Dropdown */}
                            <AnimatePresence>
                                {searchOpen && searchResults.length > 0 && (
                                    <motion.div
                                        initial={{ opacity: 0, y: 6 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        exit={{ opacity: 0, y: 6 }}
                                        className="absolute left-0 right-0 mt-3 bg-white border-2 border-gray-200 rounded-2xl shadow-xl overflow-hidden z-20"
                                    >
                                        {searchResults.map((u) => (
                                            <button
                                                key={u.id_usuario}
                                                onClick={() => handleSelectUser(u)}
                                                className="w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors flex items-center gap-3"
                                            >
                                                <div className="p-2 rounded-xl bg-gray-100">
                                                    <User className="w-4 h-4 text-gray-700" />
                                                </div>
                                                <div className="flex-1">
                                                    <div className="font-semibold text-gray-800">{u.nombre}</div>
                                                    <div className="text-xs text-gray-500">
                                                        {u.edad != null ? `${u.edad} años` : "—"} · {u.ocupacion || "—"} · {u.personalidad || "—"}
                                                    </div>
                                                </div>
                                            </button>
                                        ))}
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>

                        {searchError && (
                            <div className="mt-4 p-4 bg-red-50 border-2 border-red-200 rounded-xl">
                                <p className="text-sm text-red-700">{searchError}</p>
                            </div>
                        )}

                        {/* Expandible panel */}
                        <AnimatePresence>
                            {(isLoadingStats || userStats || statsError) && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: "auto" }}
                                    exit={{ opacity: 0, height: 0 }}
                                    className="mt-6 overflow-hidden"
                                >
                                    {/* Loader */}
                                    {isLoadingStats && (
                                        <div className="p-6 bg-gray-50 rounded-2xl border-2 border-gray-200 flex items-center gap-3">
                                            <Loader2 className="w-5 h-5 animate-spin text-gray-700" />
                                            <p className="text-gray-700 font-medium">
                                                Cargando estadísticas del usuario...
                                            </p>
                                        </div>
                                    )}

                                    {/* Error */}
                                    {statsError && (
                                        <div className="p-6 bg-red-50 rounded-2xl border-2 border-red-200">
                                            <p className="text-red-700 font-medium">{statsError}</p>
                                        </div>
                                    )}

                                    {/* Stats */}
                                    {userStats && !isLoadingStats && (
                                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                            {/* Left: user + activity cards */}
                                            <div className="lg:col-span-2 space-y-6">
                                                <div className="bg-white rounded-2xl shadow-lg p-6 border-2 border-gray-100">
                                                    <div className="flex items-start justify-between gap-4">
                                                        <div>
                                                            <h3 className="text-2xl font-bold text-gray-800">
                                                                {userStats?.user?.nombre}
                                                            </h3>
                                                            <p className="text-sm text-gray-600 mt-1">
                                                                {userStats?.user?.ocupacion || "—"} · {userStats?.user?.personalidad || "—"}
                                                            </p>
                                                            <p className="text-sm text-gray-500 mt-1">
                                                                {userStats?.user?.sexo || "—"} · {userStats?.user?.edad != null ? `${userStats.user.edad} años` : "—"} · Actividad: {userStats?.user?.p_actividad ?? "—"}
                                                            </p>
                                                        </div>

                                                        {topEmotion && (
                                                            <div className="px-4 py-3 rounded-2xl bg-gray-50 border-2 border-gray-200 text-right">
                                                                <div className="text-sm text-gray-600">Emoción dominante</div>
                                                                <div className="text-lg font-bold text-gray-800 flex items-center justify-end gap-2 mt-1">
                                                                    <span>{EMOTION_EMOJIS[topEmotion]}</span>
                                                                    <span>{formatEmotion(topEmotion)}</span>
                                                                </div>
                                                                <div className="text-sm text-gray-600 mt-1">
                                                                    {(topShare * 100).toFixed(1)}%
                                                                </div>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>

                                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                                    <StatCard
                                                        title="Entradas totales"
                                                        value={userStats?.stats?.n_entries ?? 0}
                                                    />
                                                    <StatCard
                                                        title="Tasa de actividad"
                                                        value={(userStats?.stats?.activity_rate ?? 0).toFixed(3)}
                                                    />
                                                    <StatCard
                                                        title="Entradas por semana"
                                                        value={(userStats?.stats?.entries_per_week ?? 0).toFixed(2)}
                                                    />
                                                </div>

                                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                                    <StatCard
                                                        title="Diversidad emocional"
                                                        value={userStats?.stats?.diversidad_emocional ?? 0}
                                                    />
                                                    <StatCard
                                                        title="Emoción dominante"
                                                        value={`${((userStats?.stats?.top_emotion_share ?? 0) * 100).toFixed(1)}%`}
                                                    />
                                                </div>
                                            </div>

                                            {/* Right: emotion chart */}
                                            <div className="lg:col-span-1">
                                                <EmotionChart scores={emotionScoresForChart} type="pie" />
                                            </div>
                                        </div>
                                    )}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </motion.div>
            </div>
        </section>
    );
};

const StatCard = ({ title, value }) => (
    <div className="bg-white rounded-2xl shadow-lg p-5 border-2 border-gray-100">
        <div className="text-xs uppercase tracking-wide text-gray-500">{title}</div>
        <div className="text-2xl font-bold text-gray-800 mt-2">{value}</div>
    </div>
);

export default UsersSection;
