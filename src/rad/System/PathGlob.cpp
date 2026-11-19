#include <rad/System/FileSystem.h>

#include <rad/Core/Platform.h>

#include <boost/locale/utf.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <system_error>
#include <utility>

#if defined(RAD_OS_WINDOWS)
#include <windows.h>
#endif

namespace rad
{
namespace
{

enum class TokenKind
{
    Literal,
    Any,
    Star,
    Bracket,
};

struct Token
{
    TokenKind kind;
    char32_t literal = 0;
    bool negated = false;
    std::vector<std::pair<char32_t, char32_t>> ranges{};

    [[nodiscard]] bool Matches(char32_t value) const
    {
        switch (kind)
        {
        case TokenKind::Literal:
            return literal == value;
        case TokenKind::Any:
            return true;
        case TokenKind::Bracket:
            return std::ranges::any_of(ranges, [value](const auto& range)
                                       { return range.first <= value && value <= range.second; }) !=
                   negated;
        case TokenKind::Star:
            return false; // Handled by the segment matcher.
        }
        return false;
    }
};

struct Segment
{
    bool recursive = false;
    std::vector<Token> tokens;

    [[nodiscard]] bool Matches(std::u32string_view name, bool includeDotFiles) const
    {
        if (name.starts_with(U'.') && !includeDotFiles &&
            (recursive || tokens.front().kind != TokenKind::Literal ||
             tokens.front().literal != U'.'))
        {
            return false;
        }
        if (recursive)
        {
            return true;
        }

        // Let the last '*' absorb one more code point when the following tokens fail.
        std::size_t next = 0;
        std::size_t index = 0;
        std::size_t afterStar = 0;
        std::size_t retry = 0;
        while (index < name.size())
        {
            if (next < tokens.size() && tokens[next].kind == TokenKind::Star)
            {
                afterStar = ++next;
                retry = index;
            }
            else if (next < tokens.size() && tokens[next].Matches(name[index]))
            {
                ++next;
                ++index;
            }
            else if (afterStar != 0)
            {
                next = afterStar;
                index = ++retry;
            }
            else
            {
                return false;
            }
        }
        return next == tokens.size() ||
               (next + 1 == tokens.size() && tokens[next].kind == TokenKind::Star);
    }
};

template <typename Char>
[[nodiscard]] bool Decode(std::basic_string_view<Char> input, std::u32string& output)
{
    using namespace boost::locale::utf;
    output.clear();
    auto it = input.begin();
    while (it != input.end())
    {
        const auto value = utf_traits<Char>::decode(it, input.end());
        if (value == illegal || value == incomplete)
        {
            return false;
        }
        output.push_back(static_cast<char32_t>(value));
    }
    return true;
}

[[nodiscard]] char32_t ReadLiteral(std::u32string_view& text)
{
    const auto value = text.front();
    text.remove_prefix(1);
    if (value != U'\\')
    {
        return value;
    }
    if (text.empty())
    {
        return 0;
    }
    const auto escaped = text.front();
    text.remove_prefix(1);
    return escaped;
}

[[nodiscard]] bool ParseBracket(std::u32string_view& text, Token& token)
{
    if (text.starts_with(U'!'))
    {
        token.negated = true;
        text.remove_prefix(1);
    }
    while (!text.empty())
    {
        if (text.starts_with(U']') && !token.ranges.empty())
        {
            text.remove_prefix(1);
            return true;
        }
        // POSIX character classes, equivalence classes and collating symbols are out of scope.
        if (text.starts_with(U"[:") || text.starts_with(U"[=") || text.starts_with(U"[."))
        {
            return false;
        }
        const auto first = ReadLiteral(text);
        if (first == 0)
        {
            return false;
        }
        auto last = first;
        if (text.starts_with(U'-') && text.size() > 1 && text[1] != U']')
        {
            text.remove_prefix(1);
            last = ReadLiteral(text);
            if (last == 0 || last > 0x7f || first > last)
            {
                return false;
            }
        }
        token.ranges.emplace_back(first, last);
    }
    return false;
}

[[nodiscard]] bool ParseSegment(std::u32string_view text, Segment& segment)
{
    segment.recursive = text == U"**";
    if (segment.recursive)
    {
        return true;
    }
    while (!text.empty())
    {
        switch (text.front())
        {
        case U'*':
            segment.tokens.push_back({TokenKind::Star});
            text.remove_prefix(std::min(text.find_first_not_of(U'*'), text.size()));
            break;
        case U'?':
            segment.tokens.push_back({TokenKind::Any});
            text.remove_prefix(1);
            break;
        case U'[':
            text.remove_prefix(1);
            segment.tokens.push_back({TokenKind::Bracket});
            if (!ParseBracket(text, segment.tokens.back()))
            {
                return false;
            }
            break;
        default:
            const auto value = ReadLiteral(text);
            if (value == 0)
            {
                return false;
            }
            segment.tokens.push_back({TokenKind::Literal, value});
            break;
        }
    }
    return true;
}

[[nodiscard]] bool ParsePattern(std::u32string_view text, std::vector<Segment>& pattern)
{
    if (text.starts_with(U'/'))
    {
        return false;
    }
    while (!text.empty())
    {
        const auto separator = text.find(U'/');
        const auto part = text.substr(0, separator);
        text.remove_prefix(std::min(separator, text.size()));
        if (!text.empty())
        {
            text.remove_prefix(1);
        }
        if (part.empty() || (part == U"." && pattern.empty()))
        {
            continue;
        }

        Segment segment;
        if (!ParseSegment(part, segment))
        {
            return false;
        }
        const auto& tokens = segment.tokens;
        const auto isDot = [](const Token& token)
        { return token.kind == TokenKind::Literal && token.literal == U'.'; };
        if ((tokens.size() == 1 && isDot(tokens[0])) ||
            (tokens.size() == 2 && isDot(tokens[0]) && isDot(tokens[1])))
        {
            return false;
        }
        if (pattern.empty() && tokens.size() >= 2 && tokens[0].kind == TokenKind::Literal &&
            tokens[1].kind == TokenKind::Literal && tokens[1].literal == U':')
        {
            const auto first = tokens[0].literal;
            if ((U'A' <= first && first <= U'Z') || (U'a' <= first && first <= U'z'))
            {
                return false;
            }
        }
        if (!segment.recursive || pattern.empty() || !pattern.back().recursive)
        {
            pattern.push_back(std::move(segment));
        }
    }
    return !pattern.empty();
}

void AddState(const std::vector<Segment>& pattern, std::vector<std::size_t>& states,
              std::size_t position)
{
    // Transitions arrive in order. A globstar can also be skipped without consuming a name.
    if (states.empty() || states.back() < position)
    {
        states.push_back(position);
        if (position < pattern.size() && pattern[position].recursive)
        {
            states.push_back(position + 1);
        }
    }
}

[[nodiscard]] Result<bool> IsTraversableDirectory(const std::filesystem::directory_entry& entry)
{
#if defined(RAD_OS_WINDOWS)
    const auto attributes = GetFileAttributesW(entry.path().c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES)
    {
        return Failure(std::error_code(static_cast<int>(GetLastError()), std::system_category()));
    }
    return (attributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) ==
           FILE_ATTRIBUTE_DIRECTORY;
#else
    std::error_code error;
    const auto status = entry.symlink_status(error);
    if (error)
    {
        return Failure(error);
    }
    return std::filesystem::is_directory(status);
#endif
}

struct DirectoryWork
{
    FilePath relative;
    std::vector<std::size_t> states;
};

struct Match
{
    std::u8string sortKey;
    FilePath path;
};

} // namespace

Result<std::vector<FilePath>> PathGlob(const FilePath& root, std::string_view pattern,
                                       const PathGlobOptions& options)
{
    if (root.empty() || root.native().find(FilePath::value_type{}) != FilePath::string_type::npos ||
        pattern.empty() || pattern.find('\0') != std::string_view::npos)
    {
        return Failure(std::errc::invalid_argument);
    }
    std::u32string decoded;
    if (!Decode(pattern, decoded))
    {
        return Failure(std::errc::illegal_byte_sequence);
    }
    std::vector<Segment> segments;
    if (!ParsePattern(decoded, segments))
    {
        return Failure(std::errc::invalid_argument);
    }
    const bool directoriesOnly = pattern.ends_with('/');
    const auto end = segments.size();

    std::error_code error;
    const auto status = std::filesystem::status(root, error);
    if (error || !std::filesystem::exists(status))
    {
        return Failure(error ? error : std::make_error_code(std::errc::no_such_file_or_directory));
    }
    if (!std::filesystem::is_directory(status))
    {
        return Failure(std::errc::not_a_directory);
    }

    std::vector<DirectoryWork> pending(1);
    AddState(segments, pending.back().states, 0);
    std::vector<std::size_t> next;
    std::vector<Match> matches;
    while (!pending.empty())
    {
        auto work = std::move(pending.back());
        pending.pop_back();
        const auto directory = work.relative.empty() ? root : root / work.relative;
        std::filesystem::directory_iterator it(directory, error);
        for (; !error && it != std::filesystem::directory_iterator{}; it.increment(error))
        {
            const auto name = it->path().filename();
            if (!Decode<FilePath::value_type>(name.native(), decoded))
            {
                return Failure(std::errc::illegal_byte_sequence);
            }
            bool matchesDirectly = false;
            next.clear();
            for (const auto position : work.states)
            {
                if (position == end)
                {
                    break;
                }
                const auto& segment = segments[position];
                if (segment.Matches(decoded, options.includeDotFiles))
                {
                    matchesDirectly |= position + 1 == end;
                    AddState(segments, next, segment.recursive ? position : position + 1);
                }
            }
            if (next.empty())
            {
                continue;
            }

            const bool canMatch = next.back() == end;
            const bool canContinue = next.front() != end;
            bool isDirectory = false;
            if (canContinue || directoriesOnly)
            {
                const auto result = IsTraversableDirectory(*it);
                if (!result)
                {
                    return Failure(result.error());
                }
                isDirectory = result.value();
            }
            // In 'dir/**', the zero-level match of '**' requires 'dir' to be a directory.
            if (canMatch && (isDirectory || (matchesDirectly && !directoriesOnly)))
            {
                auto path = work.relative / name;
                matches.push_back({path.generic_u8string(), std::move(path)});
            }
            if (canContinue && isDirectory)
            {
                pending.push_back({work.relative / name, next});
            }
        }
        if (error)
        {
            return Failure(error);
        }
    }

    std::ranges::sort(matches, {}, &Match::sortKey);
    std::vector<FilePath> paths;
    paths.reserve(matches.size());
    for (auto& match : matches)
    {
        paths.push_back(std::move(match.path));
    }
    return paths;
}

} // namespace rad
