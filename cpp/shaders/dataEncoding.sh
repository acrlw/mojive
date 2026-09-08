vec4 packWords(vec2 words)
{
    words = floor(words + vec2(0.5, 0.5));
    return vec4(mod(words.x, 256.0), floor(words.x / 256.0),
                mod(words.y, 256.0), floor(words.y / 256.0)) / 255.0;
}
