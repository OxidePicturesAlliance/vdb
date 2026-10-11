// Writes the Blosc test vectors in tests/data/blosc (frames produced by the C library,
// plus the raw input). Build: gcc blosc_vectors.c -o blosc_vectors -lblosc -lm
// Run: ./blosc_vectors ../blosc
#include <blosc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static const char* outdir;
static void dump(const char* name, const char* comp, int typesize, int n, int bs, int clevel, int shuffle) {
  float* src = malloc(n + 4);
  for (int i = 0; i < n/4; ++i) src[i] = (i % 50 < 25) ? 1.0f : sinf(i*0.05f) * 3.0f;
  if (n % 4) memset(((char*)src) + (n/4)*4, 7, n%4);
  char* dst = malloc(n + 16);
  int out = blosc_compress_ctx(clevel, shuffle, typesize, n, src, dst, n + 16, comp, bs, 1);
  unsigned char* h = (unsigned char*)dst;
  printf("%s %s ts=%d n=%d bs=%d -> %d : ver=%d lz=%d flags=0x%x ts=%d nbytes=%u blocksize=%u ctbytes=%u\n",
    name, comp, typesize, n, bs, out, h[0], h[1], h[2], h[3], *(unsigned*)(h+4), *(unsigned*)(h+8), *(unsigned*)(h+12));
  char fn[512]; sprintf(fn, "%s/%s.blosc", outdir, name);
  FILE* f = fopen(fn, "wb"); fwrite(dst, 1, out, f); fclose(f);
  sprintf(fn, "%s/%s.raw", outdir, name);
  f = fopen(fn, "wb"); fwrite(src, 1, n, f); fclose(f);
}
int main(int argc, char** argv) {
  outdir = argc > 1 ? argv[1] : ".";
  dump("lz4_2048", "lz4", 4, 2048, 2048, 9, 1);
  dump("lz4_64", "lz4", 4, 64, 64, 9, 1);
  dump("lz4_multiblock", "lz4", 4, 6000, 256, 9, 1);
  dump("blosclz_2048", "blosclz", 4, 2048, 2048, 9, 1);
  dump("blosclz_multiblock", "blosclz", 4, 6002, 512, 5, 1);
  dump("lz4_odd", "lz4", 2, 1026, 1026, 9, 1);
  dump("lz4_noshuf_ts1", "lz4", 1, 4096, 4096, 9, 0);
  return 0;
}
